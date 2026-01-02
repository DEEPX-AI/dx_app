#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cstdlib>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>  // STL vector container

#include "deeplabv3_postprocess.h"
#include "yolov7_postprocess.h"

// Generate color for each class ID using predefined colors for segmentation
cv::Scalar get_cityscapes_class_color(int class_id) {
    // Use predefined colors for better visualization of semantic classes
    // Cityscapes color palette for urban scene segmentation
    std::vector<cv::Scalar> colors = {
        cv::Scalar(128, 64, 128),   // road
        cv::Scalar(244, 35, 232),   // sidewalk
        cv::Scalar(70, 70, 70),     // building
        cv::Scalar(102, 102, 156),  // wall
        cv::Scalar(190, 153, 153),  // fence
        cv::Scalar(153, 153, 153),  // pole
        cv::Scalar(250, 170, 30),   // traffic light
        cv::Scalar(220, 220, 0),    // traffic sign
        cv::Scalar(107, 142, 35),   // vegetation
        cv::Scalar(152, 251, 152),  // terrain
        cv::Scalar(70, 130, 180),   // sky
        cv::Scalar(220, 20, 60),    // person
        cv::Scalar(255, 0, 0),      // rider
        cv::Scalar(0, 0, 142),      // car
        cv::Scalar(0, 0, 70),       // truck
        cv::Scalar(0, 60, 100),     // bus
        cv::Scalar(0, 80, 100),     // train
        cv::Scalar(0, 0, 230),      // motorcycle
        cv::Scalar(119, 11, 32)     // bicycle
    };

    if (class_id >= 0 && class_id < static_cast<int>(colors.size())) {
        return colors[class_id];
    }

    // Fallback to black for unknown classes
    return cv::Scalar(0, 0, 0);
}

cv::Scalar get_coco_class_color(int class_id) {
    // Use class_id as seed for consistent color generation
    // This ensures same class always gets same color
    std::srand(class_id);

    // Generate random BGR values (0-255)
    int b = std::rand() % 256;
    int g = std::rand() % 256;
    int r = std::rand() % 256;

    return cv::Scalar(b, g, r);
}

void make_letterbox_image(const cv::Mat& image, cv::Mat& preprocessed_image, const int color_space,
                          std::vector<int>& pad_xy) {
    int input_width = preprocessed_image.cols;
    int input_height = preprocessed_image.rows;
    int letterbox_pad_x = pad_xy[0];
    int letterbox_pad_y = pad_xy[1];

    cv::Mat resized_image;
    if (letterbox_pad_x == 0 && letterbox_pad_y == 0) {
        cv::resize(image, resized_image, cv::Size(input_width, input_height));
        cv::cvtColor(resized_image, preprocessed_image, color_space);
        return;
    }

    int new_width = input_width - letterbox_pad_x * 2;
    int new_height = input_height - letterbox_pad_y * 2;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));
    cv::cvtColor(resized_image, resized_image, color_space);
    int top = letterbox_pad_y;
    int bottom = letterbox_pad_y;
    int left = letterbox_pad_x;
    int right = letterbox_pad_x;

    cv::copyMakeBorder(resized_image, preprocessed_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114));
}

cv::Mat scale_segmentation_mask(const cv::Mat& mask, int orig_width, int orig_height,
                                const std::vector<int>& pad_xy) {
    // Remove letterbox padding
    int unpad_w = mask.cols - 2 * pad_xy[0];
    int unpad_h = mask.rows - 2 * pad_xy[1];

    cv::Mat unpadded_mask;
    if (pad_xy[0] > 0 || pad_xy[1] > 0) {
        cv::Rect crop_region(pad_xy[0], pad_xy[1], unpad_w, unpad_h);
        unpadded_mask = mask(crop_region).clone();
    } else {
        unpadded_mask = mask.clone();
    }

    // Resize to original image dimensions
    cv::Mat resized_mask;
    cv::resize(unpadded_mask, resized_mask, cv::Size(orig_width, orig_height), 0, 0,
               cv::INTER_NEAREST);

    return resized_mask;
}

void scale_coordinates(YOLOv7Result& detection, const std::vector<int>& pad_xy,
                       const float letterbox_scale) {
    detection.box[0] = (detection.box[0] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[1] = (detection.box[1] - static_cast<float>(pad_xy[1])) / letterbox_scale;
    detection.box[2] = (detection.box[2] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[3] = (detection.box[3] - static_cast<float>(pad_xy[1])) / letterbox_scale;
}
/**
 * @brief Visualize segmentation results by overlaying colored mask on the image.
 * @param frame Original image
 * @param segmentation_result Segmentation result containing mask
 * @param pad_xy [x, y] vector for padding size
 * @param letterbox_scale Scale factor used for letterbox
 * @param alpha Blending factor for overlay (0.0 = original image, 1.0 = only mask)
 * @return Visualized image (Mat)
 */
cv::Mat draw_segmentation(const cv::Mat& frame, const DeepLabv3Result& segmentation_result,
                          const std::vector<int>& pad_xy, const float alpha = 0.6f) {
    cv::Mat result = frame.clone();

    if (segmentation_result.segmentation_mask.empty() || segmentation_result.width == 0 ||
        segmentation_result.height == 0) {
        return result;
    }

    // Create mask image from segmentation result
    cv::Mat mask_image =
        cv::Mat::zeros(segmentation_result.height, segmentation_result.width, CV_8UC3);

    // Fill mask with class colors
    for (int y = 0; y < segmentation_result.height; ++y) {
        for (int x = 0; x < segmentation_result.width; ++x) {
            int idx = y * segmentation_result.width + x;
            if (idx < static_cast<int>(segmentation_result.segmentation_mask.size())) {
                int class_id = segmentation_result.segmentation_mask[idx];
                cv::Scalar color = get_cityscapes_class_color(class_id);
                mask_image.at<cv::Vec3b>(y, x) =
                    cv::Vec3b(static_cast<uchar>(color[0]), static_cast<uchar>(color[1]),
                              static_cast<uchar>(color[2]));
            }
        }
    }

    // Scale mask to match original frame size
    cv::Mat scaled_mask = scale_segmentation_mask(mask_image, frame.cols, frame.rows, pad_xy);

    // Blend the mask with original image
    cv::addWeighted(result, 1.0 - alpha, scaled_mask, alpha, 0, result);

    return result;
}
cv::Mat draw_detections(const cv::Mat& frame, std::vector<YOLOv7Result>& detections,
                        const std::vector<int>& pad_xy, const float letterbox_scale) {
    cv::Mat result = frame.clone();

    for (auto& detection : detections) {
        scale_coordinates(detection, pad_xy, letterbox_scale);
        // Get class-specific color
        cv::Scalar box_color = get_coco_class_color(detection.class_id);

        // Draw bounding box with class-specific color
        cv::Point2f tl(detection.box[0], detection.box[1]);
        cv::Point2f br(detection.box[2], detection.box[3]);
        cv::rectangle(result, tl, br, box_color, 2);

        // Draw class name and confidence score with background
        std::string conf_text = detection.class_name + ": " +
                                std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

        // Get text size to create background rectangle
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 1;
        int baseline = 0;
        cv::Size text_size =
            cv::getTextSize(conf_text, font_face, font_scale, thickness, &baseline);

        // Calculate text position
        cv::Point text_pos(detection.box[0], detection.box[1] - 10);

        // Draw black background rectangle
        cv::Point bg_tl(text_pos.x, text_pos.y - text_size.height);
        cv::Point bg_br(text_pos.x + text_size.width, text_pos.y + baseline);
        cv::rectangle(result, bg_tl, bg_br, cv::Scalar(0, 0, 0),
                      -1);  // Black background

        // Draw white text on black background
        cv::putText(result, conf_text, text_pos, font_face, font_scale, cv::Scalar(255, 255, 255),
                    thickness);
    }

    return result;
}

// Profiling metrics structure for Sync
struct ProfilingMetrics {
    double sum_read = 0.0;
    // YOLOv7 metrics
    double sum_yolo_preprocess = 0.0;
    double sum_yolo_inference = 0.0;
    double sum_yolo_postprocess = 0.0;

    // DeepLabV3 metrics
    double sum_deeplab_preprocess = 0.0;
    double sum_deeplab_inference = 0.0;
    double sum_deeplab_postprocess = 0.0;

    // Combined metrics
    double sum_render = 0.0;
};

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec, bool display_on) {

    auto safe_avg = [&](double sum) {
        return (total_frames > 0) ? sum / total_frames : 0.0;
    };

    double avg_read = safe_avg(metrics.sum_read);
    double avg_yolo_pre = safe_avg(metrics.sum_yolo_preprocess);
    double avg_yolo_inf = safe_avg(metrics.sum_yolo_inference);
    double avg_yolo_post = safe_avg(metrics.sum_yolo_postprocess);
    double avg_deeplab_pre = safe_avg(metrics.sum_deeplab_preprocess);
    double avg_deeplab_inf = safe_avg(metrics.sum_deeplab_inference);
    double avg_deeplab_post = safe_avg(metrics.sum_deeplab_postprocess);

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double yolo_pre_fps = avg_yolo_pre > 0 ? 1000.0 / avg_yolo_pre : 0.0;
    double yolo_inf_fps = avg_yolo_inf > 0 ? 1000.0 / avg_yolo_inf : 0.0;
    double yolo_post_fps = avg_yolo_post > 0 ? 1000.0 / avg_yolo_post : 0.0;
    double deeplab_pre_fps = avg_deeplab_pre > 0 ? 1000.0 / avg_deeplab_pre : 0.0;
    double deeplab_inf_fps = avg_deeplab_inf > 0 ? 1000.0 / avg_deeplab_inf : 0.0;
    double deeplab_post_fps = avg_deeplab_post > 0 ? 1000.0 / avg_deeplab_post : 0.0;

    auto print_model_block = [&](const std::string& name, double avg_pre, double pre_fps,
                                 double avg_inf, double inf_fps, double avg_post,
                                 double post_fps) {
        std::cout << " " << name << " Metrics" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << " " << std::left << std::setw(15) << "Preprocess" << std::right
                  << std::setw(8) << std::fixed << std::setprecision(2) << avg_pre << " ms     "
                  << std::setw(6) << std::setprecision(1) << pre_fps << " FPS" << std::endl;
        std::cout << " " << std::left << std::setw(15) << "Inference" << std::right
                  << std::setw(8) << std::fixed << std::setprecision(2) << avg_inf << " ms     "
                  << std::setw(6) << std::setprecision(1) << inf_fps << " FPS" << std::endl;
        std::cout << " " << std::left << std::setw(15) << "Postprocess" << std::right
                  << std::setw(8) << std::fixed << std::setprecision(2) << avg_post << " ms     "
                  << std::setw(6) << std::setprecision(1) << post_fps << " FPS" << std::endl;
    };

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Read" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_read << " ms     " << std::setw(6)
              << std::setprecision(1) << read_fps << " FPS" << std::endl;

    if (display_on) {
        double avg_render = metrics.sum_render / total_frames;
        double render_fps = avg_render > 0 ? 1000.0 / avg_render : 0.0;
        std::cout << " " << std::left << std::setw(15) << "Display" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(2) << avg_render << " ms     " << std::setw(6)
                  << std::setprecision(1) << render_fps << " FPS" << std::endl;
    }

    std::cout << "\n";
    print_model_block("YOLOv7", avg_yolo_pre, yolo_pre_fps, avg_yolo_inf, yolo_inf_fps,
                       avg_yolo_post, yolo_post_fps);
    std::cout << "--------------------------------------------------" << std::endl;
    print_model_block("DeepLabV3", avg_deeplab_pre, deeplab_pre_fps, avg_deeplab_inf,
                       deeplab_inf_fps, avg_deeplab_post, deeplab_post_fps);
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

    std::string yoloModelPath = "";
    std::string deeplabModelPath = "";
    std::string imgFile = "", videoFile = "", rtspUrl = "";
    int cameraIndex = -1;
    bool fps_only = false, saveVideo = false;
    int loopTest = 1, processCount = 0;

    std::string app_name = "YOLOv7 + DeepLabv3 Multi-Model Async Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("y, yolo_model", "YOLOv7 object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(yoloModelPath))(
        "d, deeplab_model", "DeepLabv3 segmentation model file (.dxnn, required)",
        cxxopts::value<std::string>(deeplabModelPath))(
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
    if (yoloModelPath.empty()) {
        std::cerr << "[ERROR] YOLOv7 model path is required. Use -y or "
                     "--yolo_model option."
                  << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    if (deeplabModelPath.empty()) {
        std::cerr << "[ERROR] DeepLabv3 model path is required. Use -d or "
                     "--deeplab_model option."
                  << std::endl;
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
    // std::cout << "=== YOLOV9 Post-Processing Sync Example ===" <<
    // std::endl;
    dxrt::InferenceOption io;
    dxrt::InferenceEngine yolo_ie(yoloModelPath, io);
    dxrt::InferenceEngine deeplabv3_ie(deeplabModelPath, io);
    
    if (!dxapp::common::minversionforRTandCompiler(&yolo_ie) ||
        !dxapp::common::minversionforRTandCompiler(&deeplabv3_ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the "
                     "version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    // Get model input dimensions (assuming both models have same input size)
    auto yolo_input_shape = yolo_ie.GetInputs().front().shape();
    auto deeplab_input_shape = deeplabv3_ie.GetInputs().front().shape();
    
    int yolo_input_height = static_cast<int>(yolo_input_shape[1]);
    int yolo_input_width = static_cast<int>(yolo_input_shape[2]);
    int deeplab_input_height = static_cast<int>(deeplab_input_shape[1]);
    int deeplab_input_width = static_cast<int>(deeplab_input_shape[2]);

    auto yolo_post_processor = YOLOv7PostProcess(
        yolo_input_width, yolo_input_height, 0.25f, 0.25f, 
        0.45f, yolo_ie.IsOrtConfigured());
    auto deeplabv3_post_processor = DeepLabv3PostProcess(deeplab_input_width, deeplab_input_height);

    
    std::cout << "[INFO] Yolo Model loaded: " << yoloModelPath << std::endl;
    std::cout << "[INFO] Model input size (WxH): " << yolo_input_width << "x" << yolo_input_width
              << std::endl;
    std::cout << std::endl;

    
    std::cout << "[INFO] Deeplab Model loaded: " << deeplabModelPath << std::endl;
    std::cout << "[INFO] Model input size (WxH): " << deeplab_input_width << "x" << deeplab_input_width
              << std::endl;
    std::cout << std::endl;

    ProfilingMetrics profiling_metrics;

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

        std::vector<int> yolo_pad_xy{0, 0};
        std::vector<int> deeplabv3_pad_xy{0, 0};
        float yolo_scale_factor = 1.f;

        yolo_scale_factor = std::min(yolo_post_processor.get_input_width() / static_cast<float>(image.cols),
                                yolo_post_processor.get_input_height() / static_cast<float>(image.rows));
        int yolo_letterbox_pad_x =
            std::max(0.f, (yolo_post_processor.get_input_width() - image.cols * yolo_scale_factor) / 2);
        int yolo_letterbox_pad_y =
            std::max(0.f, (yolo_post_processor.get_input_height() - image.rows * yolo_scale_factor) / 2);
        yolo_pad_xy = {yolo_letterbox_pad_x, yolo_letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();
        do {
            cv::Mat yolo_preprocessed_image(yolo_post_processor.get_input_height(),
                                       yolo_post_processor.get_input_width(), CV_8UC3);
            cv::Mat deeplabv3_preprocessed_image(deeplabv3_post_processor.get_input_height(),
                                       deeplabv3_post_processor.get_input_width(), CV_8UC3);
            auto t0 = std::chrono::high_resolution_clock::now();
            make_letterbox_image(image, yolo_preprocessed_image, cv::COLOR_BGR2RGB, yolo_pad_xy);
            auto t1 = std::chrono::high_resolution_clock::now();
            make_letterbox_image(image, deeplabv3_preprocessed_image, cv::COLOR_BGR2RGB, deeplabv3_pad_xy);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto outputs = yolo_ie.Run(yolo_preprocessed_image.data);
            auto t3 = std::chrono::high_resolution_clock::now();
            auto deeplabv3_outputs = deeplabv3_ie.Run(deeplabv3_preprocessed_image.data);
            auto t4 = std::chrono::high_resolution_clock::now();
            if (!outputs.empty()) {
                auto detections = yolo_post_processor.postprocess(outputs);
                auto t5 = std::chrono::high_resolution_clock::now();

                auto segmentation = deeplabv3_post_processor.postprocess(deeplabv3_outputs);
                auto t6 = std::chrono::high_resolution_clock::now();

                auto segmentation_result = draw_segmentation(image, segmentation, deeplabv3_pad_xy);
                auto t7 = std::chrono::high_resolution_clock::now();

                auto result_image = draw_detections(segmentation_result, detections, yolo_pad_xy, yolo_scale_factor);
                auto t8 = std::chrono::high_resolution_clock::now();

                if (loopTest == 1) {
                    cv::imwrite("result.jpg", result_image);
                }
                
                profiling_metrics.sum_yolo_preprocess += 
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                profiling_metrics.sum_deeplab_preprocess += 
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
                profiling_metrics.sum_yolo_inference += 
                    std::chrono::duration<double, std::milli>(t3 - t2).count();
                profiling_metrics.sum_deeplab_inference += 
                    std::chrono::duration<double, std::milli>(t4 - t3).count();
                profiling_metrics.sum_yolo_postprocess += 
                    std::chrono::duration<double, std::milli>(t5 - t4).count();
                profiling_metrics.sum_deeplab_postprocess += 
                    std::chrono::duration<double, std::milli>(t6 - t5).count();
                profiling_metrics.sum_render += 
                    std::chrono::duration<double, std::milli>(t8 - t7).count();

            }
            processCount++;
        } while (--loopTest);
        auto e = std::chrono::high_resolution_clock::now();
        
        double total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;
        print_performance_summary(profiling_metrics, processCount, total_time, !fps_only);
    } else {
        cv::VideoCapture video;
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

        std::vector<int> yolo_pad_xy{0, 0};
        std::vector<int> deeplabv3_pad_xy{0, 0};
        float yolo_scale_factor = 1.f;

        yolo_scale_factor = std::min(yolo_post_processor.get_input_width() / static_cast<float>(frame_width),
                                yolo_post_processor.get_input_height() / static_cast<float>(frame_height));
        int yolo_letterbox_pad_x =
            std::max(0.f, (yolo_post_processor.get_input_width() - frame_width * yolo_scale_factor) / 2);
        int yolo_letterbox_pad_y =
            std::max(0.f, (yolo_post_processor.get_input_height() - frame_height * yolo_scale_factor) / 2);
        yolo_pad_xy = {yolo_letterbox_pad_x, yolo_letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();
        do {
            cv::Mat image;
            auto t_read_start = std::chrono::high_resolution_clock::now();
            video >> image;
            auto t_read_end = std::chrono::high_resolution_clock::now();

            cv::Mat yolo_preprocessed_image(yolo_post_processor.get_input_height(),
                                       yolo_post_processor.get_input_width(), CV_8UC3);
            cv::Mat deeplabv3_preprocessed_image(deeplabv3_post_processor.get_input_height(),
                                       deeplabv3_post_processor.get_input_width(), CV_8UC3);
            auto t0 = std::chrono::high_resolution_clock::now();
            make_letterbox_image(image, yolo_preprocessed_image, cv::COLOR_BGR2RGB, yolo_pad_xy);
            auto t1 = std::chrono::high_resolution_clock::now();
            make_letterbox_image(image, deeplabv3_preprocessed_image, cv::COLOR_BGR2RGB, deeplabv3_pad_xy);
            auto t2 = std::chrono::high_resolution_clock::now();
            
            auto outputs = yolo_ie.Run(yolo_preprocessed_image.data);
            auto t3 = std::chrono::high_resolution_clock::now();

            auto deeplabv3_outputs = deeplabv3_ie.Run(deeplabv3_preprocessed_image.data);
            auto t4 = std::chrono::high_resolution_clock::now();

            if (!outputs.empty()) {
                auto detections = yolo_post_processor.postprocess(outputs);
                auto t5 = std::chrono::high_resolution_clock::now();
                auto segmentation = deeplabv3_post_processor.postprocess(deeplabv3_outputs);
                auto t6 = std::chrono::high_resolution_clock::now();
                auto segmentation_result = draw_segmentation(image, segmentation, deeplabv3_pad_xy);
                auto t7 = std::chrono::high_resolution_clock::now();
                auto result_image = draw_detections(segmentation_result, detections, yolo_pad_xy, yolo_scale_factor);
                auto t8 = std::chrono::high_resolution_clock::now();
                
                if (saveVideo) {
                    writer << result_image;
                }
                
                int key = -1;
                if (!fps_only) {
                    cv::imshow("result", result_image);
                    key = cv::waitKey(1);
                }
                processCount++;

                profiling_metrics.sum_yolo_preprocess += 
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                profiling_metrics.sum_deeplab_preprocess += 
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
                profiling_metrics.sum_yolo_inference += 
                    std::chrono::duration<double, std::milli>(t3 - t2).count();
                profiling_metrics.sum_deeplab_inference += 
                    std::chrono::duration<double, std::milli>(t4 - t3).count();
                profiling_metrics.sum_yolo_postprocess += 
                    std::chrono::duration<double, std::milli>(t5 - t4).count();
                profiling_metrics.sum_deeplab_postprocess += 
                    std::chrono::duration<double, std::milli>(t6 - t5).count();
                profiling_metrics.sum_render += 
                    std::chrono::duration<double, std::milli>(t8 - t7).count();

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
