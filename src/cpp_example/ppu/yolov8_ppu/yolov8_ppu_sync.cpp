#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cstdlib>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>  // STL vector container

#include "yolov8_ppu_postprocess.h"

/**
 * @brief Synchronous post-processing example for YOLOv8n object detection
model.
 *
 * - Supports image, video, and camera input sources.
 * - Performs post-processing on model inference results (decoding, NMS,
coordinate transformation, object detection, etc.).
 * - Visualization and result saving are available using OpenCV.
 * - Command-line options allow configuration of model path, input files, loop
count, FPS measurement, and result saving.
 *
 * Variable declarations and main logic are written for maintainability and code
optimization.
 */

// Generate color for each class ID using random seed
cv::Scalar get_class_color(int class_id) {
    // Use class_id as seed for consistent color generation
    // This ensures same class always gets same color
    std::srand(class_id);

    // Generate random BGR values (0-255)
    int b = std::rand() % 256;
    int g = std::rand() % 256;
    int r = std::rand() % 256;

    return cv::Scalar(b, g, r);
}

/**
 * @brief Resize the input image to the specified size and apply letterbox
 * padding for preprocessing.
 * @param image Original input image
 * @param preprocessed_image Mat object to store the preprocessed result
 * (already sized)
 * @param color_space Color space conversion code (e.g., cv::COLOR_BGR2RGB)
 * @param pad_xy [x, y] vector for padding size
 */
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

/**
 * @brief Convert detection coordinates from letterbox padded/scaled image back
 * to original image coordinates.
 * @param detection Detection result object to convert
 * @param pad_xy [x, y] vector for padding size
 * @param letterbox_scale Scale factor used for letterbox
 */
void scale_coordinates(YOLOv8_PPUResult& detection, const std::vector<int>& pad_xy,
                       const float letterbox_scale) {
    detection.box[0] = (detection.box[0] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[1] = (detection.box[1] - static_cast<float>(pad_xy[1])) / letterbox_scale;
    detection.box[2] = (detection.box[2] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[3] = (detection.box[3] - static_cast<float>(pad_xy[1])) / letterbox_scale;
}

/**
 * @brief Visualize detection results on the image by drawing bounding boxes,
 * confidence scores.
 * @param frame Original image
 * @param detections Vector of detection results
 * @param pad_xy [x, y] vector for padding size
 * @param letterbox_scale Scale factor used for letterbox
 * @return Visualized image (Mat)
 */
cv::Mat draw_detections(const cv::Mat& frame, std::vector<YOLOv8_PPUResult>& detections,
                        const std::vector<int>& pad_xy, const float letterbox_scale) {
    cv::Mat result = frame.clone();

    for (auto& detection : detections) {
        scale_coordinates(detection, pad_xy, letterbox_scale);
        // Get class-specific color
        cv::Scalar box_color = get_class_color(detection.class_id);

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

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN

    std::string modelPath = "", imgFile = "", videoFile = "";
    bool cameraMode = false, fps_only = false, saveVideo = false;
    int loopTest = 1, processCount = 0;

    std::string app_name = "YOLOv8n Post-Processing Sync Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "i, image_path", "input image file path(jpg, png, jpeg ..., required)",
        cxxopts::value<std::string>(imgFile))("v, video_path",
                                              "input video file path(mp4, mov, avi ..., required)",
                                              cxxopts::value<std::string>(videoFile))(
        "c, camera_mode", "enable camera mode",
        cxxopts::value<bool>(cameraMode)->default_value("false"))(
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
    if (!cameraMode && imgFile.empty() && videoFile.empty()) {
        std::cerr << "[ERROR] Image path or video path is required. Use -i or "
                     "--image_path option or -v or --video_path option."
                  << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    // std::cout << "=== YOLOV8n Post-Processing Sync Example ===" <<
    // std::endl;
    dxrt::InferenceOption io;
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
    auto post_processor = YOLOv8_PPUPostProcess(input_width, input_height, 0.3f, 0.45f);

    if (!imgFile.empty()) {
        cv::Mat image = cv::imread(imgFile, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Image file is not valid." << std::endl;
            exit(1);
        }

        std::vector<int> pad_xy{0, 0};
        float scale_factor = 1.f;

        scale_factor = std::min(post_processor.get_input_width() / static_cast<float>(image.cols),
                                post_processor.get_input_height() / static_cast<float>(image.rows));
        int letterbox_pad_x =
            std::max(0.f, (post_processor.get_input_width() - image.cols * scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (post_processor.get_input_height() - image.rows * scale_factor) / 2);
        pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();
        do {
            cv::Mat preprocessed_image(post_processor.get_input_height(),
                                       post_processor.get_input_width(), CV_8UC3);
            make_letterbox_image(image, preprocessed_image, cv::COLOR_BGR2RGB, pad_xy);
            auto outputs = ie.Run(preprocessed_image.data);
            if (!outputs.empty()) {
                // aligned tensor processing is now handled inside postprocess
                std::vector<YOLOv8_PPUResult> detections;
                try {
                    detections = post_processor.postprocess(outputs);
                } catch (const std::exception& e) {
                    std::cerr << "[DXAPP] [ER] Exception during postprocessing: \n"
                              << e.what() << std::endl;
                    break;
                }
                auto result_image = draw_detections(image, detections, pad_xy, scale_factor);
                if (loopTest == 1 && saveVideo) {
                    cv::imwrite("result.jpg", result_image);
                }
                if(!fps_only) cv::imshow("result", result_image);
            }
            processCount++;
        } while (--loopTest);
        auto e = std::chrono::high_resolution_clock::now();
        if (processCount == 0) {
            std::cerr << "[DXAPP] [ER] No frames were processed. Please verify the input image and "
                         "model, and ensure preprocessing/postprocessing succeeded."
                      << std::endl;
            std::cerr << "[DXAPP] [ER] Exiting." << std::endl;
            return -1;
        }
        std::cout << "[DXAPP] [INFO] total time : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us"
                  << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
                         processCount
                  << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : "
                  << processCount /
                         (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() /
                          1000.0)
                  << std::endl;
    } else if (!videoFile.empty()) {
        std::cout << "loopTest is set to 1 when a video file is provided." << std::endl;
        loopTest = 1;
        cv::VideoCapture video(videoFile);
        cv::VideoWriter writer;
        if (!video.isOpened()) {
            std::cerr << "[ERROR] Video file is not valid." << std::endl;
            exit(1);
        }

        int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (fps_only) {
            std::cout << "Processing video stream... Only FPS will be displayed." << std::endl;
        }
        if (saveVideo) {
            writer = cv::VideoWriter("result.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                     video.get(cv::CAP_PROP_FPS),
                                     cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH),
                                              video.get(cv::CAP_PROP_FRAME_HEIGHT)));
            if (!writer.isOpened()) {
                std::cerr << "[ERROR] Failed to open video writer." << std::endl;
                exit(1);
            }
        }

        std::vector<int> pad_xy{0, 0};
        float scale_factor = 1.f;

        scale_factor =
            std::min(post_processor.get_input_width() / static_cast<float>(frame_width),
                     post_processor.get_input_height() / static_cast<float>(frame_height));
        int letterbox_pad_x =
            std::max(0.f, (post_processor.get_input_width() - frame_width * scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (post_processor.get_input_height() - frame_height * scale_factor) / 2);
        pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();
        do {
            cv::Mat image;
            video >> image;
            if (image.empty()) {
                break;
            }
            cv::Mat preprocessed_image(post_processor.get_input_height(),
                                       post_processor.get_input_width(), CV_8UC3);
            make_letterbox_image(image, preprocessed_image, cv::COLOR_BGR2RGB, pad_xy);
            auto outputs = ie.Run(preprocessed_image.data);
            if (!outputs.empty()) {
                // aligned tensor processing is now handled inside postprocess
                std::vector<YOLOv8_PPUResult> detections;
                try {
                    detections = post_processor.postprocess(outputs);
                } catch (const std::exception& e) {
                    std::cerr << "[DXAPP] [ER] Exception during postprocessing: \n"
                              << e.what() << std::endl;
                    break;
                }
                auto result_image = draw_detections(image, detections, pad_xy, scale_factor);
                processCount++;
                if (saveVideo) {
                    writer << result_image;
                }
                if (fps_only) {
                    continue;
                }
                cv::imshow("result", result_image);
                if (cv::waitKey(1) == 'q') {
                    break;
                }
            }
        } while (true);
        auto e = std::chrono::high_resolution_clock::now();
        if (processCount == 0) {
            std::cerr << "[DXAPP] [ER] No frames were processed. Please verify the input image and "
                         "model, and ensure preprocessing/postprocessing succeeded."
                      << std::endl;
            std::cerr << "[DXAPP] [ER] Exiting." << std::endl;
            return -1;
        }
        std::cout << "[DXAPP] [INFO] total time : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us"
                  << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
                         processCount
                  << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : "
                  << processCount /
                         (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() /
                          1000.0)
                  << std::endl;
    }

    std::cout << "\nExample completed successfully!" << std::endl;
    DXRT_TRY_CATCH_END
    return 0;
}
