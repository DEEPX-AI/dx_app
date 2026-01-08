#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>  // STL vector container

#include "yolov5pose_postprocess.h"

/**
 * @brief Synchronous post-processing example for YOLOV5Pose pose detection
model.
 *
 * - Supports image, video, and camera input sources.
 * - Performs post-processing on model inference results (decoding, NMS,
coordinate transformation, landmark extraction, etc.).
 * - Visualization and result saving are available using OpenCV.
 * - Command-line options allow configuration of model path, input files, loop
count, FPS measurement, and result saving.
 *
 * Variable declarations and main logic are written for maintainability and code
optimization.
 */
  
static const std::vector<std::vector<int>> skeleton = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
    {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},   {1, 3},  {2, 4},  {3, 5}, {4, 6},
};

static const std::vector<cv::Scalar> pose_limb_color = {
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(255, 51, 255), cv::Scalar(255, 51, 255),
    cv::Scalar(255, 51, 255), cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0),
};

static const std::vector<cv::Scalar> pose_kpt_color = {
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
};

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
void scale_coordinates(YOLOv5PoseResult& detection, const std::vector<int>& pad_xy,
                       const float letterbox_scale) {
    detection.box[0] = (detection.box[0] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[1] = (detection.box[1] - static_cast<float>(pad_xy[1])) / letterbox_scale;
    detection.box[2] = (detection.box[2] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[3] = (detection.box[3] - static_cast<float>(pad_xy[1])) / letterbox_scale;
    for (size_t i = 0; i < detection.landmarks.size(); i += 3) {
        detection.landmarks[i] =
            (detection.landmarks[i] - static_cast<float>(pad_xy[0])) / letterbox_scale;
        detection.landmarks[i + 1] =
            (detection.landmarks[i + 1] - static_cast<float>(pad_xy[1])) / letterbox_scale;
    }
}

/**
 * @brief Visualize detection results on the image by drawing bounding boxes,
 * confidence scores, and landmarks.
 * @param frame Original image
 * @param detections Vector of detection results
 * @param pad_xy [x, y] vector for padding size
 * @param letterbox_scale Scale factor used for letterbox
 * @return Visualized image (Mat)
 */
cv::Mat draw_detections(const cv::Mat& frame, std::vector<YOLOv5PoseResult>& detections,
                        const std::vector<int>& pad_xy, const float letterbox_scale) {
    cv::Mat result = frame.clone();

    for (auto& detection : detections) {
        scale_coordinates(detection, pad_xy, letterbox_scale);
        // Draw bounding box
        cv::Point2f tl(detection.box[0], detection.box[1]);
        cv::Point2f br(detection.box[2], detection.box[3]);
        cv::rectangle(result, tl, br, cv::Scalar(0, 255, 0), 2);

        // Draw confidence score with background
        std::string conf_text =
            "Person: " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

        // Get text size to create background rectangle
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 1;
        int baseline = 0;
        cv::Size text_size =
            cv::getTextSize(conf_text, font_face, font_scale, thickness, &baseline);

        // Calculate text position
        cv::Point text_pos(static_cast<int>(detection.box[0]),
                           static_cast<int>(detection.box[1]) - 10);

        // Draw black background rectangle
        cv::Point bg_tl(text_pos.x, text_pos.y - text_size.height);
        cv::Point bg_br(text_pos.x + text_size.width, text_pos.y + baseline);
        cv::rectangle(result, bg_tl, bg_br, cv::Scalar(0, 0, 0),
                      -1);  // Black background

        // Draw white text on black background
        cv::putText(result, conf_text, text_pos, font_face, font_scale, cv::Scalar(255, 255, 255),
                    thickness);

        // Draw landmarks if requested
        if (!detection.landmarks.empty()) {
            for (size_t i = 0; i < skeleton.size(); ++i) {
                auto& p = skeleton[i];
                if (detection.landmarks[3 * p[0]] >= 0 && detection.landmarks[3 * p[1]] >= 0 &&
                    detection.landmarks[3 * p[0] + 2] >= 0.3) {
                    cv::Point2f pt1(detection.landmarks[3 * p[0]],
                                    detection.landmarks[3 * p[0] + 1]);
                    cv::Point2f pt2(detection.landmarks[3 * p[1]],
                                    detection.landmarks[3 * p[1] + 1]);
                    cv::line(result, pt1, pt2, pose_limb_color[i], 2, cv::LINE_AA);
                }
            }

            for (size_t i = 0; i < detection.landmarks.size(); i += 3) {
                if (detection.landmarks[i + 2] < 0.3) continue;
                cv::Point2f landmark(detection.landmarks[i], detection.landmarks[i + 1]);
                cv::circle(result, landmark, 3, pose_kpt_color[i / 3], -1, cv::LINE_AA);
            }
        }
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

    std::string app_name = "YOLOV5Pose Post-Processing Sync Example";
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
    // std::cout << "=== YOLOV5Pose Post-Processing Sync Example ===" <<
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
    auto post_processor =
        YOLOv5PosePostProcess(input_width, input_height, 0.5, 0.5, 0.45, ie.IsOrtConfigured());

    std::cout << "[INFO] Model loaded: " << modelPath << std::endl;
    std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height
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
            auto t0 = std::chrono::high_resolution_clock::now();
            cv::Mat preprocessed_image(post_processor.get_input_height(),
                                       post_processor.get_input_width(), CV_8UC3);
            make_letterbox_image(image, preprocessed_image, cv::COLOR_BGR2RGB, pad_xy);
            auto t1 = std::chrono::high_resolution_clock::now();

            auto outputs = ie.Run(preprocessed_image.data);
            auto t2 = std::chrono::high_resolution_clock::now();

            if (!outputs.empty()) {
                // aligned tensor processing is now handled inside postprocess
                auto detections = post_processor.postprocess(outputs);
                auto t3 = std::chrono::high_resolution_clock::now();

                auto result_image = draw_detections(image, detections, pad_xy, scale_factor);
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
            auto t_read_start = std::chrono::high_resolution_clock::now();
            video >> image;
            auto t_read_end = std::chrono::high_resolution_clock::now();

            if (image.empty()) {
                break;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            cv::Mat preprocessed_image(post_processor.get_input_height(),
                                       post_processor.get_input_width(), CV_8UC3);
            make_letterbox_image(image, preprocessed_image, cv::COLOR_BGR2RGB, pad_xy);
            auto t1 = std::chrono::high_resolution_clock::now();

            auto outputs = ie.Run(preprocessed_image.data);
            auto t2 = std::chrono::high_resolution_clock::now();

            if (!outputs.empty()) {
                // aligned tensor processing is now handled inside postprocess
                auto detections = post_processor.postprocess(outputs);
                auto t3 = std::chrono::high_resolution_clock::now();

                auto result_image = draw_detections(image, detections, pad_xy, scale_factor);

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
