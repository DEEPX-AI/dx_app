#include <dxrt/dxrt_api.h>

#include <chrono>  // For timing
#include <common_util.hpp>
#include <cstdlib>
#include <cxxopts.hpp>
#include <iomanip>  // For std::setprecision
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>  // STL vector container

#include "yolov8seg_postprocess.h"

/**
 * @brief Synchronous post-processing example for YOLOv8-seg instance segmentation model.
 *
 * - Supports image, video, and camera input sources.
 * - Performs post-processing on model inference results (object detection + instance segmentation).
 * - Visualization and result saving are available using OpenCV.
 * - Command-line options allow configuration of model path, input files, loop
 * count, FPS measurement, and result saving.
 *
 * Variable declarations and main logic are written for maintainability and code
 * optimization.
 */

// Pre-computed color table for consistent and efficient color generation
static const std::vector<cv::Scalar> CLASS_COLORS = {
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
inline cv::Scalar get_instance_color(int class_id) {
    return CLASS_COLORS[class_id % CLASS_COLORS.size()];
}

/**
 * @brief Resize the input image to the specified size and apply letterbox
 * padding for preprocessing YOLOv8 segmentation.
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
    float scale_x = static_cast<float>(input_width) / image.cols;
    float scale_y = static_cast<float>(input_height) / image.rows;
    float scale = std::min(scale_x, scale_y);

    ratio[0] = scale;
    ratio[1] = scale;

    // Calculate new dimensions after scaling
    int new_width = static_cast<int>(image.cols * scale);
    int new_height = static_cast<int>(image.rows * scale);

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
    box[0] = (box[0] - pad_xy[0]) / ratio[0];  // x1
    box[1] = (box[1] - pad_xy[1]) / ratio[1];  // y1
    box[2] = (box[2] - pad_xy[0]) / ratio[0];  // x2
    box[3] = (box[3] - pad_xy[1]) / ratio[1];  // y2

    // Clamp to image boundaries
    box[0] = std::max(0.0f, std::min(static_cast<float>(orig_width), box[0]));
    box[1] = std::max(0.0f, std::min(static_cast<float>(orig_height), box[1]));
    box[2] = std::max(0.0f, std::min(static_cast<float>(orig_width), box[2]));
    box[3] = std::max(0.0f, std::min(static_cast<float>(orig_height), box[3]));
}

/**
 * @brief Transform segmentation mask from model input coordinates to original image coordinates.
 * @param mask Segmentation mask (flattened)
 * @param mask_width Width of the mask
 * @param mask_height Height of the mask
 * @param pad_xy Padding offsets [x, y]
 * @param ratio Scale ratios [x, y]
 * @param orig_width Original image width
 * @param orig_height Original image height
 * @return Transformed mask as cv::Mat
 */
cv::Mat transform_mask_to_original(const std::vector<float>& mask, int mask_width, int mask_height,
                                   const std::vector<int>& pad_xy, const std::vector<float>& ratio,
                                   int orig_width, int orig_height) {
    // Create mask Mat directly from data pointer for efficiency
    cv::Mat mask_mat(mask_height, mask_width, CV_32F, const_cast<float*>(mask.data()));

    // Calculate the actual content size after removing padding
    const float scale_factor = ratio[0];
    const int content_width = static_cast<int>(orig_width * scale_factor);
    const int content_height = static_cast<int>(orig_height * scale_factor);

    // Remove padding - crop to the actual content area
    cv::Mat unpadded_mask;
    if (pad_xy[0] > 0 || pad_xy[1] > 0) {
        const int crop_x = pad_xy[0];
        const int crop_y = pad_xy[1];
        const int crop_w = std::min(content_width, mask_width - 2 * pad_xy[0]);
        const int crop_h = std::min(content_height, mask_height - 2 * pad_xy[1]);

        if (crop_w > 0 && crop_h > 0 && crop_x + crop_w <= mask_width &&
            crop_y + crop_h <= mask_height) {
            cv::Rect crop_region(crop_x, crop_y, crop_w, crop_h);
            unpadded_mask = mask_mat(crop_region);
        } else {
            unpadded_mask = mask_mat;
        }
    } else {
        unpadded_mask = mask_mat;
    }

    // Scale back to original image size using more efficient interpolation
    cv::Mat original_mask;
    cv::resize(unpadded_mask, original_mask, cv::Size(orig_width, orig_height), 0, 0,
               cv::INTER_LINEAR);

    // Convert to binary mask in-place
    cv::threshold(original_mask, original_mask, 0.5f, 1.0f, cv::THRESH_BINARY);

    return original_mask;
}

/**
 * @brief Visualize YOLOv8 segmentation results by drawing bounding boxes and masks.
 * @param frame Original image
 * @param results YOLOv8 segmentation results
 * @param pad_xy Padding offsets [x, y]
 * @param ratio Scale ratios [x, y]
 * @param alpha Blending factor for mask overlay (0.0 = original image, 1.0 = only mask)
 * @return Visualized image
 */
cv::Mat draw_yolov8_segmentation(const cv::Mat& frame, const std::vector<YOLOv8SegResult>& results,
                                 const std::vector<int>& pad_xy, const std::vector<float>& ratio,
                                 const float alpha = 0.6f) {
    cv::Mat result = frame.clone();

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& detection = results[i];

        // Transform bounding box to original coordinates
        std::vector<float> box = detection.box;
        transform_box_to_original(box, pad_xy, ratio, frame.cols, frame.rows);

        // Draw bounding box
        cv::Scalar color = get_instance_color(i);
        cv::Point pt1(static_cast<int>(box[0]), static_cast<int>(box[1]));
        cv::Point pt2(static_cast<int>(box[2]), static_cast<int>(box[3]));
        cv::rectangle(result, pt1, pt2, color, 2);

        // Draw segmentation mask if available - ultra-optimized version
        if (!detection.mask.empty() && detection.mask_width > 0 && detection.mask_height > 0) {
            cv::Mat mask = transform_mask_to_original(detection.mask, detection.mask_width,
                                                      detection.mask_height, pad_xy, ratio,
                                                      frame.cols, frame.rows);

            // Pre-calculate blending constants
            const float inv_alpha = 1.0f - alpha;
            const int color_b = static_cast<int>(color[0]);
            const int color_g = static_cast<int>(color[1]);
            const int color_r = static_cast<int>(color[2]);

            // Vectorized mask application with row-wise processing
            for (int y = 0; y < mask.rows; ++y) {
                const float* mask_row = mask.ptr<float>(y);
                cv::Vec3b* result_row = result.ptr<cv::Vec3b>(y);

                // Process 4 pixels at once for better cache usage
                int x = 0;
                for (; x < mask.cols - 3; x += 4) {
                    // Check if any of the 4 pixels need processing
                    bool needs_blend[4] = {mask_row[x] > 0.5f, mask_row[x + 1] > 0.5f,
                                           mask_row[x + 2] > 0.5f, mask_row[x + 3] > 0.5f};

                    // Vectorized blending for valid pixels
                    for (int i = 0; i < 4; ++i) {
                        if (needs_blend[i]) {
                            cv::Vec3b& pixel = result_row[x + i];
                            pixel[0] = static_cast<uchar>(pixel[0] * inv_alpha + color_b * alpha);
                            pixel[1] = static_cast<uchar>(pixel[1] * inv_alpha + color_g * alpha);
                            pixel[2] = static_cast<uchar>(pixel[2] * inv_alpha + color_r * alpha);
                        }
                    }
                }

                // Handle remaining pixels
                for (; x < mask.cols; ++x) {
                    if (mask_row[x] > 0.5f) {
                        cv::Vec3b& pixel = result_row[x];
                        pixel[0] = static_cast<uchar>(pixel[0] * inv_alpha + color_b * alpha);
                        pixel[1] = static_cast<uchar>(pixel[1] * inv_alpha + color_g * alpha);
                        pixel[2] = static_cast<uchar>(pixel[2] * inv_alpha + color_r * alpha);
                    }
                }
            }
        }

        // Draw class label and confidence
        std::string label = detection.class_name + ": " +
                            std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);

        cv::Point label_pt(pt1.x, pt1.y - 10 > 10 ? pt1.y - 10 : pt1.y + label_size.height + 10);

        // Draw label background
        cv::rectangle(result, cv::Point(label_pt.x, label_pt.y - label_size.height - 5),
                      cv::Point(label_pt.x + label_size.width, label_pt.y + baseline), color,
                      cv::FILLED);

        // Draw label text
        cv::putText(result, label, label_pt, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 2);
    }

    return result;
}

// Profiling metrics structure for Sync
struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    double sum_postprocess = 0.0;
    double sum_render = 0.0;
};

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec, bool display_on) {
    if (total_frames == 0) return;

    double avg_read = metrics.sum_read / total_frames;
    double avg_pre = metrics.sum_preprocess / total_frames;
    double avg_inf = metrics.sum_inference / total_frames;
    double avg_post = metrics.sum_postprocess / total_frames;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
    double inf_fps = avg_inf > 0 ? 1000.0 / avg_inf : 0.0;
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
              << std::setprecision(1) << inf_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Postprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_post << " ms     " << std::setw(6)
              << std::setprecision(1) << post_fps << " FPS" << std::endl;

    if (display_on) {
        double avg_render = metrics.sum_render / total_frames;
        double render_fps = avg_render > 0 ? 1000.0 / avg_render : 0.0;
        std::cout << " " << std::left << std::setw(15) << "Display" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(2) << avg_render << " ms     " << std::setw(6)
                  << std::setprecision(1) << render_fps << " FPS" << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;

    double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;

    std::cout << " " << std::left << std::setw(19) << "Total Frames"
              << " :    " << total_frames << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Time"
              << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
              << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Overall FPS"
              << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
              << std::endl;
    std::cout << "==================================================" << std::endl;
}

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN

    std::string modelPath = "", imgFile = "", videoFile = "", rtspUrl = "";
    int cameraIndex = -1;
    bool fps_only = false, saveVideo = false;
    int loopTest = 1, processCount = 0;

    std::string app_name = "YOLOv8 Segmentation Post-Processing Sync Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "i, image_path", "input image file path(jpg, png, jpeg ...)",
        cxxopts::value<std::string>(imgFile))("v, video_path",
                                              "input video file path(mp4, mov, avi ...)",
                                              cxxopts::value<std::string>(videoFile))(
        "c, camera_index", "camera device index (e.g., 0)",
        cxxopts::value<int>(cameraIndex))("r, rtsp_url", "RTSP stream URL",
                                          cxxopts::value<std::string>(rtspUrl))(
        "s, save_video", "save processed video",
        cxxopts::value<bool>(saveVideo)->default_value("false"))(
        "l, loop", "Number of inference iterations to run",
        cxxopts::value<int>(loopTest)->default_value("1"))(
        "no-display", "will not visualize, only show fps",
        cxxopts::value<bool>(fps_only)->default_value("false"))("h, help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    // Validate required arguments
    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path is required. Use -m or --model_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    int sourceCount = 0;
    if (!imgFile.empty()) sourceCount++;
    if (!videoFile.empty()) sourceCount++;
    if (cameraIndex >= 0) sourceCount++;
    if (!rtspUrl.empty()) sourceCount++;

    if (sourceCount != 1) {
        std::cerr << "[ERROR] Please specify exactly one input source: image (-i), video (-v), "
                     "camera (-c), or RTSP (-r)."
                  << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    // std::cout << "=== YOLOv8-Seg Post-Processing Sync Example ===" << std::endl;

    dxrt::InferenceOption io;
    io.useORT = true;  // YOLOv8-seg requires ORT mode
    dxrt::InferenceEngine ie(modelPath, io);

    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the "
                     "version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    auto input_shape = ie.GetInputs().front().shape();
    int input_height = static_cast<int>(input_shape[1]);
    int input_width = static_cast<int>(input_shape[2]);

    // Create post-processor with stricter score and NMS thresholds to reduce false positives
    auto post_processor = YOLOv8SegPostProcess(input_width, input_height, 0.5f, 0.45f, true);

    std::cout << "[INFO] Model loaded: " << modelPath << std::endl;
    std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height
              << std::endl;
    std::cout << std::endl;

    ProfilingMetrics profiling_metrics;

    // Pre-allocate preprocessing buffer for better performance
    cv::Mat preprocessed_image(input_height, input_width, CV_8UC3);

    if (!imgFile.empty()) {
        cv::Mat image = cv::imread(imgFile, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Image file is not valid." << std::endl;
            exit(1);
        }

        std::cout << "[INFO] Image file: " << imgFile << std::endl;
        std::cout << "[INFO] Input source resolution (WxH): " << image.cols << "x" << image.rows
                  << std::endl;
        std::cout << std::endl;
        std::cout << "[INFO] Starting inference..." << std::endl;

        std::vector<float> ratio = {
            post_processor.get_input_width() / static_cast<float>(image.cols),
            post_processor.get_input_height() / static_cast<float>(image.rows)};
        auto scale_factor = std::min(ratio[0], ratio[1]);
        int letterbox_pad_x =
            std::max(0.f, (post_processor.get_input_width() - image.cols * scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (post_processor.get_input_height() - image.rows * scale_factor) / 2);
        std::vector<int> pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();
        do {
            auto t0 = std::chrono::high_resolution_clock::now();

            // Reuse pre-allocated buffer
            make_letterbox_image(image, preprocessed_image, pad_xy, ratio);
            auto t1 = std::chrono::high_resolution_clock::now();

            auto outputs = ie.Run(preprocessed_image.data);
            auto t2 = std::chrono::high_resolution_clock::now();

            if (!outputs.empty()) {
                auto segmentation_results = post_processor.postprocess(outputs);
                auto t3 = std::chrono::high_resolution_clock::now();

                cv::Mat result_image =
                    draw_yolov8_segmentation(image, segmentation_results, pad_xy, ratio);
                auto t4 = std::chrono::high_resolution_clock::now();
                
                if (loopTest == 1 && saveVideo) {
                    cv::imwrite("result.jpg", result_image);
                }
                if(!fps_only){
                    cv::imshow("result", result_image); 
                    std::ignore = cv::waitKey(1);
                } 

                profiling_metrics.sum_preprocess +=
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                profiling_metrics.sum_inference +=
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
                profiling_metrics.sum_postprocess +=
                    std::chrono::duration<double, std::milli>(t3 - t2).count();
                profiling_metrics.sum_render +=
                    std::chrono::duration<double, std::milli>(t4 - t3).count();
            }
            processCount++;
        } while (--loopTest);
        auto e = std::chrono::high_resolution_clock::now();double total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;

        print_performance_summary(profiling_metrics, processCount, total_time, !fps_only);
    } else {cv::VideoCapture video;
        std::string source_info;
        bool is_file = !videoFile.empty();

        if (cameraIndex >= 0) {
            video.open(cameraIndex);
            source_info = "Camera index: " + std::to_string(cameraIndex);
        } else if (!rtspUrl.empty()) {
            video.open(rtspUrl);
            source_info = "RTSP URL: " + rtspUrl;
        } else {
            video.open(videoFile);
            source_info = "Video file: " + videoFile;
            std::cout << "loopTest is set to 1 when a video file is provided." << std::endl;
            loopTest = 1;
        }
        
        cv::VideoWriter writer;
        if (!video.isOpened()) {
            std::cerr << "[ERROR] Failed to open input source." << std::endl;
            exit(1);
        }

        int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = video.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "[INFO] " << source_info << std::endl;
        std::cout << "[INFO] Input source resolution (WxH): " << frame_width << "x" << frame_height
                  << std::endl;
        std::cout << "[INFO] Input source FPS: " << std::fixed << std::setprecision(2) << fps
                  << std::endl;
        if (is_file) {
            std::cout << "[INFO] Total frames: " << total_frames << std::endl;
        }
        std::cout << std::endl;
        std::cout << "[INFO] Starting inference..." << std::endl;

        if (fps_only) {
            std::cout << "Processing video stream... Only FPS will be displayed." << std::endl;
        }
        if (saveVideo) {
            writer = cv::VideoWriter("result.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                     fps > 0 ? fps : 30.0,
                                     cv::Size(frame_width, frame_height));
            if (!writer.isOpened()) {
                std::cerr << "[ERROR] Failed to open video writer." << std::endl;
                exit(1);
            }
        }

        std::vector<int> pad_xy(2, 0);
        std::vector<float> ratio(2, 1.0f);

        ratio = {post_processor.get_input_width() / static_cast<float>(frame_width),
                 post_processor.get_input_height() / static_cast<float>(frame_height)};
        auto scale_factor = std::min(ratio[0], ratio[1]);
        int letterbox_pad_x =
            std::max(0.f, (post_processor.get_input_width() - frame_width * scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (post_processor.get_input_height() - frame_height * scale_factor) / 2);
        pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();
        do {
            cv::Mat image;
            auto t_read_start = std::chrono::high_resolution_clock::now();
            video >> image;
            auto t_read_end = std::chrono::high_resolution_clock::now();

            if (image.empty()) {
                break;
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            // Reuse pre-allocated buffer
            make_letterbox_image(image, preprocessed_image, pad_xy, ratio);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto outputs = ie.Run(preprocessed_image.data);
            auto t2 = std::chrono::high_resolution_clock::now();
            if (!outputs.empty()) {
                auto segmentation_results = post_processor.postprocess(outputs);
                auto t3 = std::chrono::high_resolution_clock::now();

                auto result_image =
                    draw_yolov8_segmentation(image, segmentation_results, pad_xy, ratio);
                    
                if (saveVideo) {
                    writer << result_image;
                }
                int key = -1;
                if (!fps_only) {
                    cv::imshow("result", result_image);
                    key = cv::waitKey(1);
                }
                auto t4 = std::chrono::high_resolution_clock::now();

                processCount++;

                profiling_metrics.sum_read +=
                    std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
                profiling_metrics.sum_preprocess +=
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                profiling_metrics.sum_inference +=
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
                profiling_metrics.sum_postprocess +=
                    std::chrono::duration<double, std::milli>(t3 - t2).count();
                profiling_metrics.sum_render +=
                    std::chrono::duration<double, std::milli>(t4 - t3).count();

                if (key == 'q') {
                    break;
                }
            }
        } while (true);
        auto e = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;

        print_performance_summary(profiling_metrics, processCount, total_time, !fps_only);
    }

    std::cout << "\nExample completed successfully!" << std::endl;
    DXRT_TRY_CATCH_END
    return 0;
}
