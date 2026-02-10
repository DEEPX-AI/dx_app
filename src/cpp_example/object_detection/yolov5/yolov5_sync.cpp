#include <dxrt/dxrt_api.h>

#include <chrono>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "yolov5_postprocess.h"

/**
 * @brief Synchronous post-processing example for YOLOv5 object detection model.
 *
 * - Supports image, video, and camera input sources.
 * - Performs post-processing on model inference results (decoding, NMS,
 * coordinate transformation, object detection, etc.).
 * - Visualization and result saving are available using OpenCV.
 * - Command-line options allow configuration of model path, input files, loop
 * count, FPS measurement, and result saving.
 *
 * Variable declarations and main logic are written for maintainability and code
 * optimization.
 */


constexpr size_t SHOW_WINDOW_SIZE_W = 960;
constexpr size_t SHOW_WINDOW_SIZE_H = 640;

// --- Structures ---

// Profiling metrics structure
struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    double sum_postprocess = 0.0;
    double sum_render = 0.0;
    int infer_completed = 0;
};

// Command line arguments structure
struct CommandLineArgs {
    std::string modelPath;
    std::string imageFilePath;
    std::string videoFile;
    std::string rtspUrl;
    int cameraIndex = -1;
    bool no_display = false;
    bool saveVideo = false;
    int loopTest = -1;
};

// Pre-computed color table for class visualization (optimized for performance)
static const std::vector<cv::Scalar> COCO_CLASS_COLORS = {
    cv::Scalar(255, 0, 0),      // Red
    cv::Scalar(0, 255, 0),      // Green
    cv::Scalar(0, 0, 255),      // Blue
    cv::Scalar(255, 255, 0),    // Cyan
    cv::Scalar(255, 0, 255),    // Magenta
    cv::Scalar(0, 255, 255),    // Yellow
    cv::Scalar(128, 0, 128),    // Purple
    cv::Scalar(255, 165, 0),    // Orange
    cv::Scalar(0, 128, 0),      // Dark Green
    cv::Scalar(128, 128, 0),    // Olive
    cv::Scalar(0, 128, 128),    // Teal
    cv::Scalar(128, 0, 0),      // Maroon
    cv::Scalar(192, 192, 192),  // Silver
    cv::Scalar(255, 192, 203),  // Pink
    cv::Scalar(255, 215, 0),    // Gold
    cv::Scalar(173, 216, 230),  // Light Blue
    cv::Scalar(144, 238, 144),  // Light Green
    cv::Scalar(255, 218, 185),  // Peach
    cv::Scalar(221, 160, 221),  // Plum
    cv::Scalar(255, 240, 245)   // Lavender Blush
};

// Generate color for each class ID using pre-computed table
inline cv::Scalar get_cityscapes_class_color(int class_id) {
    return COCO_CLASS_COLORS[class_id % COCO_CLASS_COLORS.size()];
}

// --- Helper functions for postprocessing ---

bool handle_postprocess_exception(const std::exception& e, const std::string& context) {
    std::cerr << "[DXAPP] [ER] " << context << " error during postprocessing: \n"
              << e.what() << std::endl;
    return false;
}

bool process_postprocess(YOLOv5PostProcess& post_processor, const std::vector<std::shared_ptr<dxrt::Tensor>>& outputs, std::vector<YOLOv5Result>& detections_vec) {
    try {
        detections_vec = post_processor.postprocess(outputs);
        return true;
    } catch (const std::invalid_argument& e) {
        return handle_postprocess_exception(e, "Invalid argument");
    } catch (const std::out_of_range& e) {
        return handle_postprocess_exception(e, "Out of range");
    } catch (const std::length_error& e) {
        return handle_postprocess_exception(e, "Length");
    } catch (const std::domain_error& e) {
        return handle_postprocess_exception(e, "Domain");
    } catch (const std::range_error& e) {
        return handle_postprocess_exception(e, "Range");
    } catch (const std::overflow_error& e) {
        return handle_postprocess_exception(e, "Overflow");
    } catch (const std::underflow_error& e) {
        return handle_postprocess_exception(e, "Underflow");
    }
}

// --- Other helper functions ---

/**
 * @brief Check if file extension indicates an image file.
 */
bool is_image_file(const std::string& extension) {
    return extension == ".jpg" || extension == ".jpeg" || 
           extension == ".png" || extension == ".bmp";
}

/**
 * @brief Load image files from a directory.
 */
std::vector<std::string> load_image_files_from_directory(const std::string& dirPath) {
    std::vector<std::string> imageFiles;
    
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!fs::is_regular_file(entry.path())) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (is_image_file(ext)) {
            imageFiles.push_back(entry.path().string());
        }
    }
    std::sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

/**
 * @brief Process image path (file or directory) and return list of image files.
 */
std::pair<std::vector<std::string>, int> process_image_path(
    const std::string& imageFilePath, int loopTest) {
    std::vector<std::string> imageFiles;
    
    if (fs::is_directory(imageFilePath)) {
        imageFiles = load_image_files_from_directory(imageFilePath);
        if (imageFiles.empty()) {
            std::cerr << "[ERROR] No image files found in directory: " << imageFilePath << std::endl;
            exit(1);
        }
        if (loopTest == -1) {
            loopTest = static_cast<int>(imageFiles.size());
        }
    } else if (fs::is_regular_file(imageFilePath)) {
        imageFiles.push_back(imageFilePath);
        if (loopTest == -1) {
            loopTest = 1;
        }
    } else {
        std::cerr << "[ERROR] Invalid image path: " << imageFilePath << std::endl;
        exit(1);
    }
    
    return {imageFiles, loopTest};
}

// --- Callback helper functions ---

/**
 * @brief Resize the input image to the specified size and apply letterbox
 * padding for preprocessing.
 * @param image Original input image
 * @param preprocessed_image Mat object to store the preprocessed result
 * @param pad_xy [x, y] vector for padding size
 * @param ratio [x, y] vector for scale ratio
 */
void make_letterbox_image(const cv::Mat& image, cv::Mat& preprocessed_image,
                          std::vector<int>& pad_xy, std::vector<float>& ratio) {
    int input_width = preprocessed_image.cols;
    int input_height = preprocessed_image.rows;

    // Calculate scale ratio
    float scale_x = static_cast<float>(input_width) / static_cast<float>(image.cols);
    float scale_y = static_cast<float>(input_height) / static_cast<float>(image.rows);
    float scale = std::min(scale_x, scale_y);

    ratio[0] = scale;
    ratio[1] = scale;

    // Calculate new dimensions after scaling
    auto new_width = static_cast<int>(static_cast<float>(image.cols) * scale);
    auto new_height = static_cast<int>(static_cast<float>(image.rows) * scale);

    // Calculate padding
    pad_xy[0] = (input_width - new_width) / 2;
    pad_xy[1] = (input_height - new_height) / 2;

    // Resize image
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));

    // Convert BGR to RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    // Apply padding
    int top = pad_xy[1];
    int bottom = input_height - new_height - top;
    int left = pad_xy[0];
    int right = input_width - new_width - left;

    cv::copyMakeBorder(resized_image, preprocessed_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

/**
 * @brief Transform detection boxes from model input coordinates to original image coordinates.
 * @param box Bounding box coordinates [x1, y1, x2, y2]
 * @param pad_xy Padding offsets [x, y]
 * @param ratio Scale ratios [x, y]
 * @param orig_width Original image width
 * @param orig_height Original image height
 */
void transform_box_to_original(std::vector<float>& box, const std::vector<int>& pad_xy,
                               const std::vector<float>& ratio, int orig_width, int orig_height) {
    // Remove padding and scale back to original coordinates
    box[0] = (box[0] - static_cast<float>(pad_xy[0])) / ratio[0];  // x1
    box[1] = (box[1] - static_cast<float>(pad_xy[1])) / ratio[1];  // y1
    box[2] = (box[2] - static_cast<float>(pad_xy[0])) / ratio[0];  // x2
    box[3] = (box[3] - static_cast<float>(pad_xy[1])) / ratio[1];  // y2

    // Clamp to image boundaries
    box[0] = std::max(0.0f, std::min(static_cast<float>(orig_width), box[0]));
    box[1] = std::max(0.0f, std::min(static_cast<float>(orig_height), box[1]));
    box[2] = std::max(0.0f, std::min(static_cast<float>(orig_width), box[2]));
    box[3] = std::max(0.0f, std::min(static_cast<float>(orig_height), box[3]));
}

/**
 * @brief Draw a single detection bounding box with label on the image.
 * @param result Image to draw on (modified in place)
 * @param box Box coordinates [x1, y1, x2, y2]
 * @param detection Detection result containing class info
 * @param color Color for the bounding box
 */
void draw_detection_box(cv::Mat& result, const std::vector<float>& box,
                        const YOLOv5Result& detection, const cv::Scalar& color) {
    cv::Point pt1(static_cast<int>(box[0]), static_cast<int>(box[1]));
    cv::Point pt2(static_cast<int>(box[2]), static_cast<int>(box[3]));
    cv::rectangle(result, pt1, pt2, color, 2);

    std::string label = detection.class_name + ": " +
                        std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);

    cv::Point label_pt(pt1.x, pt1.y - 10 > 10 ? pt1.y - 10 : pt1.y + label_size.height + 10);

    cv::rectangle(result, cv::Point(label_pt.x, label_pt.y - label_size.height - 5),
                  cv::Point(label_pt.x + label_size.width, label_pt.y + baseline), color,
                  cv::FILLED);

    cv::putText(result, label, label_pt, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 2);
}

/**
 * @brief Visualize detection results on the image by drawing bounding boxes,
 * confidence scores.
 * @param frame Original image
 * @param detections Vector of detection results
 * @param pad_xy [x, y] vector for padding size
 * @param ratio [x, y] vector for scale ratio
 * @return Visualized image (Mat)
 */
cv::Mat draw_detections(const cv::Mat& frame, const std::vector<YOLOv5Result>& detections,
                        const std::vector<int>& pad_xy, const std::vector<float>& ratio) {
    cv::Mat result = frame.clone();

    for (const auto& detection : detections) {
        // Transform bounding box to original coordinates
        std::vector<float> box = detection.box;
        transform_box_to_original(box, pad_xy, ratio, frame.cols, frame.rows);

        // Get class-specific color
        cv::Scalar color = get_cityscapes_class_color(detection.class_id);
        draw_detection_box(result, box, detection, color);
    }

    return result;
}

// --- Performance summary ---

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec, bool display_on) {
    if (metrics.infer_completed == 0) return;

    double avg_read = metrics.sum_read / metrics.infer_completed;
    double avg_pre = metrics.sum_preprocess / metrics.infer_completed;
    double avg_inf = metrics.sum_inference / metrics.infer_completed;
    double avg_post = metrics.sum_postprocess / metrics.infer_completed;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
    double infer_fps = avg_inf > 0 ? 1000.0 / avg_inf : 0.0;
    double post_fps = avg_post > 0 ? 1000.0 / avg_post : 0.0;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Read" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_read << " ms     " << std::setw(6)
              << std::setprecision(1) << read_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Preprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_pre << " ms     " << std::setw(6)
              << std::setprecision(1) << pre_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Inference" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_inf << " ms     " << std::setw(6)
              << std::setprecision(1) << infer_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Postprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_post << " ms     " << std::setw(6)
              << std::setprecision(1) << post_fps << " FPS" << std::endl;

    if (display_on) {
        double avg_render = metrics.sum_render / metrics.infer_completed;
        double render_fps = avg_render > 0 ? 1000.0 / avg_render : 0.0;
        std::cout << " " << std::left << std::setw(15) << "Display" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(2) << avg_render << " ms     " << std::setw(6)
                  << std::setprecision(1) << render_fps << " FPS" << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Frames"
              << " :    " << total_frames << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Time"
              << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
              << std::endl;

    double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;
    std::cout << " " << std::left << std::setw(19) << "Overall FPS"
              << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
              << std::endl;
    std::cout << "==================================================" << std::endl;
}

// --- Command line parsing and validation ---

// Parse and validate command line arguments
CommandLineArgs parse_command_line(int argc, char* argv[]) {
    CommandLineArgs args;
    std::string app_name = "YOLOv5 Post-Processing Sync Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(args.modelPath))(
        "i, image_path", "input image file path or directory containing images (supports jpg, png, jpeg, bmp)",
        cxxopts::value<std::string>(args.imageFilePath))("v, video_path",
                                              "input video file path(mp4, mov, avi ...)",
                                              cxxopts::value<std::string>(args.videoFile))(
        "c, camera_index", "camera device index (e.g., 0)",
        cxxopts::value<int>(args.cameraIndex))("r, rtsp_url", "RTSP stream URL",
                                          cxxopts::value<std::string>(args.rtspUrl))(
        "s, save_video", "save processed video",
        cxxopts::value<bool>(args.saveVideo)->default_value("false"))(
        "l, loop", "Number of inference iterations to run",
        cxxopts::value<int>(args.loopTest)->default_value("-1"))(
        "no-display", "will not visualize, only show fps",
        cxxopts::value<bool>(args.no_display)->default_value("false"))("h, help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    return args;
}

// Validate command line arguments
void validate_arguments(const CommandLineArgs& args) {
    if (args.modelPath.empty()) {
        std::cerr << "[ERROR] Model path is required. Use -m or --model_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    int sourceCount = 0;
    if (!args.imageFilePath.empty()) sourceCount++;
    if (!args.videoFile.empty()) sourceCount++;
    if (args.cameraIndex >= 0) sourceCount++;
    if (!args.rtspUrl.empty()) sourceCount++;

    if (sourceCount != 1) {
        std::cerr << "[ERROR] Please specify exactly one input source: image (-i), video (-v), "
                     "camera (-c), or RTSP (-r)." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
}

// Open video capture based on input source
bool open_video_capture(cv::VideoCapture& video, const CommandLineArgs& args) {
    if (args.cameraIndex >= 0) {
        video.open(args.cameraIndex);
    } else if (!args.rtspUrl.empty()) {
        video.open(args.rtspUrl);
    } else {
        video.open(args.videoFile);
    }
    return video.isOpened();
}

// --- Frame processing functions ---

// Process a single frame (preprocess, inference, postprocess, render)
bool process_single_frame(
    const cv::Mat& input_frame, cv::Mat& display_image, cv::Mat& preprocessed_image,
    dxrt::InferenceEngine& ie,
    YOLOv5PostProcess& post_processor, ProfilingMetrics& metrics,
    cv::VideoWriter& writer, bool no_display, bool saveVideo, double t_read) {
    
    if (input_frame.empty()) {
        std::cerr << "[ERROR] Empty input frame" << std::endl;
        return false;
    }

    std::vector<int> pad_xy{0, 0};
    std::vector<float> ratio{1.0f, 1.0f};

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::resize(input_frame, display_image, cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
    make_letterbox_image(display_image, preprocessed_image, pad_xy, ratio);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto outputs = ie.Run(preprocessed_image.data, nullptr, nullptr);
    auto t2 = std::chrono::high_resolution_clock::now();
    double t_inference = std::chrono::duration<double, std::milli>(t2 - t1).count();

    if (outputs.empty()) {
        return true;
    }

    // Postprocess
    std::vector<YOLOv5Result> detections_vec;
    if (!process_postprocess(post_processor, outputs, detections_vec)) {
        return false;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_postprocess = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Render
    auto render_start = std::chrono::high_resolution_clock::now();
    auto processed_frame = draw_detections(display_image, detections_vec, pad_xy, ratio);

    bool quit_requested = false;
    if (!processed_frame.empty()) {
        if (saveVideo) writer << processed_frame;
        if (!no_display) {
            cv::imshow("result", processed_frame);
            if (cv::waitKey(1) == 'q') quit_requested = true;
        }
    }
    auto render_end = std::chrono::high_resolution_clock::now();
    double t_render = std::chrono::duration<double, std::milli>(render_end - render_start).count();

    // Update metrics
    metrics.sum_read += t_read;
    metrics.sum_preprocess += t_preprocess;
    metrics.sum_inference += t_inference;
    metrics.sum_postprocess += t_postprocess;
    metrics.sum_render += t_render;
    metrics.infer_completed++;

    return !quit_requested;
}

// Process image frames loop
void process_image_frames(
    const std::vector<std::string>& imageFiles, const std::string& imageFilePath,
    int loopTest, cv::Mat& display_image, cv::Mat& preprocessed_image,
    dxrt::InferenceEngine& ie,
    YOLOv5PostProcess& post_processor, ProfilingMetrics& metrics,
    int& processCount, cv::VideoWriter& writer, bool no_display, bool saveVideo) {
    
    for (int i = 0; i < loopTest; ++i) {
        std::string currentImagePath = imageFiles.empty() ? imageFilePath : imageFiles[i % imageFiles.size()];
        
        auto tr0 = std::chrono::high_resolution_clock::now();
        cv::Mat img = cv::imread(currentImagePath);
        auto tr1 = std::chrono::high_resolution_clock::now();
        double t_read = std::chrono::duration<double, std::milli>(tr1 - tr0).count();

        if (img.empty()) {
            std::cerr << "[ERROR] Failed to read image: " << currentImagePath << std::endl;
            continue;
        }

        if (!process_single_frame(img, display_image, preprocessed_image,
                                  ie, post_processor, metrics, writer, no_display, saveVideo, t_read)) {
            break;
        }
        processCount++;
    }
}

// Process video frames loop
void process_video_frames(
    cv::VideoCapture& video, cv::Mat& display_image, cv::Mat& preprocessed_image,
    dxrt::InferenceEngine& ie,
    YOLOv5PostProcess& post_processor, ProfilingMetrics& metrics,
    int& processCount, cv::VideoWriter& writer, bool no_display, bool saveVideo) {
    
    bool should_continue = true;
    while (should_continue) {
        cv::Mat frame;
        auto tr0 = std::chrono::high_resolution_clock::now();
        video >> frame;
        auto tr1 = std::chrono::high_resolution_clock::now();
        double t_read = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
        
        if (frame.empty()) {
            should_continue = false;
            continue;
        }

        if (!process_single_frame(frame, display_image, preprocessed_image,
                                  ie, post_processor, metrics, writer, no_display, saveVideo, t_read)) {
            should_continue = false;
            continue;
        }
        processCount++;
    }
}

// --- Main function ---

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    int processCount = 0;

    CommandLineArgs args = parse_command_line(argc, argv);
    validate_arguments(args);

    // Handle image file or directory
    std::vector<std::string> imageFiles;
    bool is_image = !args.imageFilePath.empty();
    int loopTest = args.loopTest;
    if (is_image) {
        auto result = process_image_path(args.imageFilePath, loopTest);
        imageFiles = result.first;
        loopTest = result.second;
    } else if (loopTest == -1) {
        loopTest = 1;
    }

    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(args.modelPath, io);
    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    auto input_shape = ie.GetInputs().front().shape();
    auto input_height = static_cast<int>(input_shape[1]);
    auto input_width = static_cast<int>(input_shape[2]);
    auto post_processor = YOLOv5PostProcess(input_width, input_height, 0.25f, 0.25f, 0.45f, ie.IsOrtConfigured());

    std::cout << "[INFO] Model loaded: " << args.modelPath << std::endl;
    std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height << std::endl;
    std::cout << std::endl;

    std::vector<uint8_t> input_buffer(ie.GetInputSize());
    ProfilingMetrics profiling_metrics;

    cv::VideoCapture video;
    if (!is_image && !open_video_capture(video, args)) {
        std::cerr << "[ERROR] Failed to open input source." << std::endl;
        return -1;
    }

    cv::VideoWriter writer;

    // Update info and setup for video if needed
    if (!is_image) {
        auto frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
        auto frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = video.get(cv::CAP_PROP_FPS);
        auto total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

        std::string source_info;
        if (args.cameraIndex >= 0) {
            source_info = "Camera index: " + std::to_string(args.cameraIndex);
        } else if (!args.rtspUrl.empty()) {
            source_info = "RTSP URL: " + args.rtspUrl;
        } else {
            source_info = "Video file: " + args.videoFile;
            std::cout << "loopTest is set to 1 when a video file is provided." << std::endl;
            loopTest = 1;
        }

        std::cout << "[INFO] " << source_info << std::endl;
        std::cout << "[INFO] Input source resolution (WxH): " << frame_width << "x" << frame_height
                  << std::endl;
        std::cout << "[INFO] Input source FPS: " << std::fixed << std::setprecision(2) << fps
                  << std::endl;
        if (!args.videoFile.empty()) {
            std::cout << "[INFO] Total frames: " << total_frames << std::endl;
        }
        std::cout << std::endl;

        // Video Save Setup
        if (args.saveVideo) {
            writer.open("result.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps > 0 ? fps : 30.0,
                        cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            if (!writer.isOpened()) {
                std::cerr << "[ERROR] Failed to open video writer." << std::endl;
                exit(1);
            }
        }
    }

    std::cout << "[INFO] Starting inference..." << std::endl;
    if (args.no_display) {
        std::cout << "Processing... Only FPS will be displayed." << std::endl;
    }

    cv::Mat display_image(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3);
    cv::Mat preprocessed_image(input_height, input_width, CV_8UC3, input_buffer.data());
    auto s_time = std::chrono::high_resolution_clock::now();

    if (is_image) {
        process_image_frames(imageFiles, args.imageFilePath, loopTest, display_image, preprocessed_image,
                             ie, post_processor, profiling_metrics,
                             processCount, writer, args.no_display, args.saveVideo);
    } else {
        process_video_frames(video, display_image, preprocessed_image,
                             ie, post_processor, profiling_metrics, processCount,
                             writer, args.no_display, args.saveVideo);
    }

    auto e_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(e_time - s_time).count();
    print_performance_summary(profiling_metrics, processCount, total_time, !args.no_display);

    DXRT_TRY_CATCH_END
    return 0;
}
