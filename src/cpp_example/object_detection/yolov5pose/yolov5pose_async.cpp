#include <dxrt/dxrt_api.h>

#include <atomic>  // For thread-safe counters
#include <chrono>  // For timing measurements
#include <common_util.hpp>
#include <condition_variable>  // For thread synchronization
#include <cxxopts.hpp>
#include <iomanip>  // For std::setprecision
#include <iostream>
#include <memory>  // For smart pointers
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>  // STL vector container

#include "yolov5pose_postprocess.h"

/**
 * @brief Asynchronous post-processing example for YOLOV5Pose pose detection
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

#define ASYNC_BUFFER_SIZE 40
#define MAX_QUEUE_SIZE 100

#define SHOW_WINDOW_SIZE_W 960
#define SHOW_WINDOW_SIZE_H 640

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

// Profiling metrics structure
struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    double sum_postprocess = 0.0;
    double sum_render = 0.0;
    int infer_completed = 0;

    // Inflight tracking
    std::chrono::high_resolution_clock::time_point infer_first_ts;
    std::chrono::high_resolution_clock::time_point infer_last_ts;
    std::chrono::high_resolution_clock::time_point inflight_last_ts;
    int inflight_current = 0;
    int inflight_max = 0;
    double inflight_time_sum = 0.0;
    bool first_inference = true;

    std::mutex metrics_mutex;
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

struct DetectionArgs {
    std::vector<YOLOv5PoseResult>* detections;
    cv::Mat* current_frame;
    dxrt::InferenceEngine* ie;
    YOLOv5PosePostProcess* ypp;
    std::mutex output_postprocess_lk;
    int* processed_count = nullptr;
    int request_id = 0;
    bool is_no_show = false;
    bool is_video_save = false;

    // Timing information
    double t_preprocess = 0.0;
    std::chrono::high_resolution_clock::time_point t_run_async_start;
    ProfilingMetrics* metrics = nullptr;

    // Constructor for proper initialization
    DetectionArgs()
        : detections(nullptr),
          current_frame(nullptr),
          ie(nullptr),
          ypp(nullptr),
          processed_count(nullptr),
          request_id(0),
          is_no_show(false),
          is_video_save(false),
          t_preprocess(0.0),
          metrics(nullptr) {}

    // Destructor for proper cleanup
    ~DetectionArgs() {
        if (detections != nullptr) {
            delete detections;
            detections = nullptr;
        }
    }
};

struct DisplayArgs {
    std::vector<YOLOv5PoseResult>* detections;
    cv::Mat* original_frame;
    YOLOv5PosePostProcess* ypp;
    std::mutex display_lk;
    bool is_no_show = false;
    bool is_video_save = false;
    int* processed_count = nullptr;

    // Timing information
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    ProfilingMetrics* metrics = nullptr;

    DisplayArgs()
        : detections(nullptr),
          original_frame(nullptr),
          ypp(nullptr),
          is_no_show(false),
          is_video_save(false),
          processed_count(nullptr),
          t_preprocess(0.0),
          t_inference(0.0),
          t_postprocess(0.0),
          metrics(nullptr) {}

    ~DisplayArgs() {
        if (detections != nullptr) {
            delete detections;
            detections = nullptr;
        }
    }
};

// Thread-safe queue wrapper
class SafeQueue {
   private:
    std::queue<DetectionArgs*> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;

   public:
    SafeQueue(size_t max_size = MAX_QUEUE_SIZE) : max_size_(max_size) {}

    void push(DetectionArgs* item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return queue_.size() < max_size_; });
        queue_.push(item);
        condition_.notify_one();
    }

    DetectionArgs* pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        DetectionArgs* item = queue_.front();
        queue_.pop();
        condition_.notify_one();
        return item;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return queue_.size();
    }
};

void post_process_thread_func(SafeQueue* wait_queue, std::queue<DisplayArgs*>* display_queue,
                              std::atomic<int>* appQuit) {
    while (appQuit->load() == -1) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    std::cout << "[DXAPP] [INFO] post processing thread start" << std::endl;

    std::vector<std::unique_ptr<DisplayArgs>> display_args_list;
    display_args_list.reserve(ASYNC_BUFFER_SIZE);

    while (appQuit->load() == 0) {
        if (wait_queue->size() > 0) {
            DetectionArgs* args = wait_queue->pop();
            auto display_args = std::unique_ptr<DisplayArgs>(new DisplayArgs());

            // outputs: inference 결과 텐서 벡터
            auto outputs = args->ie->Wait(args->request_id);

            // Calculate inference time
            auto t1 = std::chrono::high_resolution_clock::now();
            double inference_time =
                std::chrono::duration<double, std::milli>(t1 - args->t_run_async_start).count();

            
            auto detections_vec = args->ypp->postprocess(outputs);

            // Calculate postprocess time
            auto t2 = std::chrono::high_resolution_clock::now();
            double postprocess_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Update inflight tracking
            if (args->metrics) {
                std::unique_lock<std::mutex> metrics_lock(args->metrics->metrics_mutex);

                args->metrics->infer_last_ts = t1;
                args->metrics->infer_completed++;

                // Update inflight tracking - decrease current count
                auto dt =
                    std::chrono::duration<double>(t1 - args->metrics->inflight_last_ts).count();
                args->metrics->inflight_time_sum += args->metrics->inflight_current * dt;
                args->metrics->inflight_last_ts = t1;
                args->metrics->inflight_current--;
            }
            // detections 멤버는 포인터이므로, 벡터를 동적할당하여 복사
            {
                std::unique_lock<std::mutex> lock(args->output_postprocess_lk);
                display_args->detections = new std::vector<YOLOv5PoseResult>(detections_vec);
                display_args->original_frame = new cv::Mat(args->current_frame->clone());
                // processed_count: number of processed frames
                display_args->processed_count = args->processed_count;
                // ypp: pointer to YoloPostProcess instance
                display_args->ypp = args->ypp;
                display_args->is_no_show = args->is_no_show;
                display_args->is_video_save = args->is_video_save;

                // Copy timing information
                display_args->t_preprocess = args->t_preprocess;
                display_args->t_inference = inference_time;
                display_args->t_postprocess = postprocess_time;
                display_args->metrics = args->metrics;

                display_queue->push(display_args.get());

                display_args_list.push_back(std::move(display_args));  // unique_ptr로 소유권 이전
            }
        }
    }
}

void display_thread_func(std::queue<DisplayArgs*>* display_queue, std::atomic<int>* appQuit,
                         cv::VideoWriter* writer, std::vector<int>* pad_xy, float* scale_factor) {
    while (appQuit->load() == -1) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    std::cout << "[DXAPP] [INFO] display thread start" << std::endl;
    while (appQuit->load() == 0) {
        if (!display_queue->empty()) {
            // DisplayArgs 포인터를 안전하게 pop 받음
            auto args = display_queue->front();
            display_queue->pop();

            // Start render timing
            auto render_start = std::chrono::high_resolution_clock::now();

            // detection 결과를 벡터로 변환하여 draw_detections에 전달
            auto processed_frame =
                draw_detections(*args->original_frame, *args->detections, *pad_xy, *scale_factor);

            {
                std::unique_lock<std::mutex> lock(args->display_lk);

                if (args->detections != nullptr) {
                    delete args->detections;
                    args->detections = nullptr;
                }
                (*args->processed_count)++;

                if (processed_frame.dims != 0) {
                    if (args->is_video_save) {
                        *writer << processed_frame;
                    }
                    if (!args->is_no_show) {
                        cv::imshow("result", processed_frame);
                        if (cv::waitKey(1) == 'q') {
                            args->is_no_show = true;
                            appQuit->store(1);
                        }
                    }
                }

                // Calculate render time
                auto render_end = std::chrono::high_resolution_clock::now();
                double render_time =
                    std::chrono::duration<double, std::milli>(render_end - render_start).count();

                // Update metrics with timing information
                if (args->metrics) {
                    std::unique_lock<std::mutex> metrics_lock(args->metrics->metrics_mutex);
                    args->metrics->sum_preprocess += args->t_preprocess;
                    args->metrics->sum_inference += args->t_inference;
                    args->metrics->sum_postprocess += args->t_postprocess;
                    args->metrics->sum_render += render_time;
                }
            }
        }
    }
}

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec, bool display_on) {
    if (metrics.infer_completed == 0) return;

    double avg_read = metrics.sum_read / metrics.infer_completed;
    double avg_pre = metrics.sum_preprocess / metrics.infer_completed;
    double avg_inf = metrics.sum_inference / metrics.infer_completed;
    double avg_post = metrics.sum_postprocess / metrics.infer_completed;

    auto inflight_time_window =
        std::chrono::duration<double>(metrics.infer_last_ts - metrics.infer_first_ts).count();

    double infer_tp =
        (inflight_time_window > 0) ? metrics.infer_completed / inflight_time_window : 0.0;
    double inflight_avg =
        (inflight_time_window > 0) ? metrics.inflight_time_sum / inflight_time_window : 0.0;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
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
              << std::setprecision(1) << infer_tp << " FPS*" << std::endl;
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
    std::cout << " * Actual throughput via async inference" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Completed"
              << " :    " << metrics.infer_completed << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Inflight Avg"
              << " :    " << std::fixed << std::setprecision(1) << inflight_avg << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Inflight Max"
              << " :      " << metrics.inflight_max << std::endl;
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
    std::atomic<int> appQuit(-1);
    std::string modelPath = "", imgFile = "", videoFile = "", rtspUrl = "";
    int cameraIndex = -1;
    bool fps_only = false, saveVideo = false;
    int loopTest = 1, loopCount = 1, processCount = 0;

    std::string app_name = "YOLOV5Pose Post-Processing Async Example";
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
        std::cerr << "[ERROR] Model path is required. Use -m or "
                     "--model_path option."
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
    loopCount = loopTest;
    // std::cout << "=== YOLOV5Pose Post-Processing ASync Example ===" <<
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
        // Print model input size
    std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height
              << std::endl;

    std::vector<std::vector<uint8_t>> input_buffers(ASYNC_BUFFER_SIZE);
    std::vector<std::vector<uint8_t>> output_buffers(ASYNC_BUFFER_SIZE);
    for (auto& input_buffer : input_buffers) {
        input_buffer = std::vector<uint8_t>(ie.GetInputSize());
    }
    for (auto& output_buffer : output_buffers) {
        output_buffer = std::vector<uint8_t>(ie.GetOutputSize());
    }

    SafeQueue wait_queue;
    std::queue<DisplayArgs*> display_queue;
    ProfilingMetrics profiling_metrics;

    std::thread post_process_thread(post_process_thread_func, &wait_queue, &display_queue,
                                    &appQuit);
    cv::VideoWriter writer;
    std::vector<int> pad_xy{0, 0};
    float scale_factor = 1.f;
    std::thread display_thread(display_thread_func, &display_queue, &appQuit, &writer, &pad_xy,
                               &scale_factor);


    if (!imgFile.empty()) {

        cv::Mat image = cv::imread(imgFile, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Image file is not valid." << std::endl;
            exit(1);
        }

        // Print original image resolution
        std::cout << "[INFO] Image resolution (WxH): " << image.cols << "x" << image.rows
                  << std::endl;
        std::cout << "[INFO] Total frames: " << loopCount << std::endl;

        cv::resize(image, image, cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H),
                   cv::INTER_LINEAR);

        scale_factor = std::min(post_processor.get_input_width() / static_cast<float>(image.cols),
                                post_processor.get_input_height() / static_cast<float>(image.rows));
        int letterbox_pad_x =
            std::max(0.f, (post_processor.get_input_width() - image.cols * scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (post_processor.get_input_height() - image.rows * scale_factor) / 2);
        pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();

        // Create DetectionArgs objects with proper lifetime management
        std::vector<std::unique_ptr<DetectionArgs>> detection_args_list;
        detection_args_list.reserve(ASYNC_BUFFER_SIZE);
        int index = 0;
        do {
            index = (index + 1) % ASYNC_BUFFER_SIZE;

            // Start preprocess timing
            auto t0 = std::chrono::high_resolution_clock::now();

            cv::Mat preprocessed_image =
                cv::Mat(post_processor.get_input_height(), post_processor.get_input_width(),
                        CV_8UC3, input_buffers[index].data());
            make_letterbox_image(image, preprocessed_image, cv::COLOR_BGR2RGB, pad_xy);

            // Calculate preprocess time and start async inference
            auto t1 = std::chrono::high_resolution_clock::now();
            double preprocess_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto req_id =
                ie.RunAsync(preprocessed_image.data, nullptr, output_buffers[index].data());
            auto t2 = std::chrono::high_resolution_clock::now();

            // Update inflight tracking
            {
                std::unique_lock<std::mutex> metrics_lock(profiling_metrics.metrics_mutex);

                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1;
                    profiling_metrics.inflight_last_ts = t2;
                    profiling_metrics.first_inference = false;
                } else {
                    auto dt = std::chrono::duration<double>(t2 - profiling_metrics.inflight_last_ts)
                                  .count();
                    profiling_metrics.inflight_time_sum += profiling_metrics.inflight_current * dt;
                    profiling_metrics.inflight_last_ts = t2;
                }

                profiling_metrics.inflight_current++;
                if (profiling_metrics.inflight_current > profiling_metrics.inflight_max) {
                    profiling_metrics.inflight_max = profiling_metrics.inflight_current;
                }
            }

            // Create DetectionArgs with proper lifetime
            auto args = std::unique_ptr<DetectionArgs>(new DetectionArgs());
            args->ie = &ie;
            args->ypp = &post_processor;
            args->current_frame = &image;
            args->detections = nullptr;
            args->request_id = req_id;
            args->processed_count = &processCount;
            args->is_no_show = fps_only;
            args->is_video_save = saveVideo;
            args->t_preprocess = preprocess_time;
            args->t_run_async_start = t1;
            args->metrics = &profiling_metrics;

            wait_queue.push(args.get());
            detection_args_list.push_back(std::move(args));
            if (appQuit.load() == -1) {
                appQuit.store(0);
            }
            if (appQuit.load() == 1) break;
        } while (--loopTest);

        // Wait for all processing to complete
        while (processCount < loopCount && appQuit.load() == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        appQuit.store(1);
        post_process_thread.join();
        display_thread.join();

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

        if (!video.isOpened()) {
            std::cerr << "[ERROR] Failed to open input source." << std::endl;
            exit(1);
        }

        // Print video information
        int video_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
        int video_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
        double video_fps = video.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "[INFO] " << source_info << std::endl;
        std::cout << "[INFO] Video resolution (WxH): " << video_width << "x" << video_height
                  << std::endl;
        std::cout << "[INFO] Video FPS: " << std::fixed << std::setprecision(2) << video_fps
                  << std::endl;
        if (is_file) {
            std::cout << "[INFO] Total frames: " << total_frames << std::endl;
            loopCount = total_frames;
        } else {
            loopCount = std::numeric_limits<int>::max();
        }

        if (fps_only) {
            std::cout << "Processing video stream... Only FPS will be displayed." << std::endl;
        }
        if (saveVideo) {
            writer = cv::VideoWriter("result.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                     video_fps > 0 ? video_fps : 30.0,
                                     cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            if (!writer.isOpened()) {
                std::cerr << "[ERROR] Failed to open video writer." << std::endl;
                exit(1);
            }
        }

        scale_factor =
            std::min(post_processor.get_input_width() / static_cast<float>(SHOW_WINDOW_SIZE_W),
                     post_processor.get_input_height() / static_cast<float>(SHOW_WINDOW_SIZE_H));
        int letterbox_pad_x = std::max(
            0.f, (post_processor.get_input_width() - SHOW_WINDOW_SIZE_W * scale_factor) / 2);
        int letterbox_pad_y = std::max(
            0.f, (post_processor.get_input_height() - SHOW_WINDOW_SIZE_H * scale_factor) / 2);
        pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();

        // Create DetectionArgs objects with proper lifetime management
        std::vector<std::unique_ptr<DetectionArgs>> detection_args_list;
        detection_args_list.reserve(ASYNC_BUFFER_SIZE);
        std::vector<cv::Mat> images(ASYNC_BUFFER_SIZE);
        for (auto& image : images) {
            image = cv::Mat(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3);
        }
        int index = 0;
        int submitted_frames = 0;
        do {
            index = (index + 1) % ASYNC_BUFFER_SIZE;
            cv::Mat frame;

            auto t_read_start = std::chrono::high_resolution_clock::now();
            video >> frame;
            auto t_read_end = std::chrono::high_resolution_clock::now();

            if (frame.empty()) {
                break;
            }
            submitted_frames++;

            {
                std::unique_lock<std::mutex> metrics_lock(profiling_metrics.metrics_mutex);
                profiling_metrics.sum_read +=
                    std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
            }

            // Start preprocess timing
            auto t0 = std::chrono::high_resolution_clock::now();

            cv::resize(frame, images[index], cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H),
                       cv::INTER_LINEAR);
            cv::Mat preprocessed_image =
                cv::Mat(post_processor.get_input_height(), post_processor.get_input_width(),
                        CV_8UC3, input_buffers[index].data());
            make_letterbox_image(images[index], preprocessed_image, cv::COLOR_BGR2RGB, pad_xy);

            // Calculate preprocess time and start async inference
            auto t1 = std::chrono::high_resolution_clock::now();
            double preprocess_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto req_id =
                ie.RunAsync(preprocessed_image.data, nullptr, output_buffers[index].data());
            auto t2 = std::chrono::high_resolution_clock::now();

            // Update inflight tracking
            {
                std::unique_lock<std::mutex> metrics_lock(profiling_metrics.metrics_mutex);

                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1;
                    profiling_metrics.inflight_last_ts = t2;
                    profiling_metrics.first_inference = false;
                } else {
                    auto dt = std::chrono::duration<double>(t2 - profiling_metrics.inflight_last_ts)
                                  .count();
                    profiling_metrics.inflight_time_sum += profiling_metrics.inflight_current * dt;
                    profiling_metrics.inflight_last_ts = t2;
                }

                profiling_metrics.inflight_current++;
                if (profiling_metrics.inflight_current > profiling_metrics.inflight_max) {
                    profiling_metrics.inflight_max = profiling_metrics.inflight_current;
                }
            }

            // Create DetectionArgs with proper lifetime
            auto args = std::unique_ptr<DetectionArgs>(new DetectionArgs());
            args->ie = &ie;
            args->ypp = &post_processor;
            args->current_frame = &images[index];
            args->detections = nullptr;
            args->request_id = req_id;
            args->processed_count = &processCount;
            args->is_no_show = fps_only;
            args->is_video_save = saveVideo;
            args->t_preprocess = preprocess_time;
            args->t_run_async_start = t1;
            args->metrics = &profiling_metrics;

            wait_queue.push(args.get());
            detection_args_list.push_back(std::move(args));
            if (appQuit.load() == -1) {
                appQuit.store(0);
            }
            if (appQuit.load() == 1) break;
        } while (true);

        loopCount = submitted_frames;

        // Wait for all processing to complete
        while (processCount < loopCount && appQuit.load() == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        appQuit.store(1);
        post_process_thread.join();
        display_thread.join();

        auto e = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;

        print_performance_summary(profiling_metrics, processCount, total_time, !fps_only);
    }

    std::cout << "\nExample completed successfully!" << std::endl;
    DXRT_TRY_CATCH_END
    return 0;
}
