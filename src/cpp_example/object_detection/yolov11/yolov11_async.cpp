#include <dxrt/dxrt_api.h>

#include <atomic>
#include <chrono>
#include <common_util.hpp>
#include <condition_variable>
#include <cxxopts.hpp>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>

#include "yolov11_postprocess.h"

/**
 * @brief Asynchronous post-processing example for YOLOv11 object detection
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

#define ASYNC_BUFFER_SIZE 40
#define MAX_QUEUE_SIZE 100

#define SHOW_WINDOW_SIZE_W 960
#define SHOW_WINDOW_SIZE_H 640

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

struct DisplayArgs {
    std::shared_ptr<std::vector<YOLOv11Result>> detections;
    std::shared_ptr<cv::Mat> original_frame;

    YOLOv11PostProcess *ypp = nullptr;
    int *processed_count = nullptr;
    bool is_no_show = false;
    bool is_video_save = false;
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    ProfilingMetrics *metrics = nullptr;

    DisplayArgs() = default;
};

struct DetectionArgs {
    cv::Mat current_frame;
    dxrt::InferenceEngine *ie = nullptr;
    YOLOv11PostProcess *ypp = nullptr;
    ProfilingMetrics *metrics = nullptr;
    int *processed_count = nullptr;
    int request_id = 0;
    bool is_no_show = false;
    bool is_video_save = false;
    double t_preprocess = 0.0;
    std::chrono::high_resolution_clock::time_point t_run_async_start;

    DetectionArgs() = default;
};

// --- 2. SafeQueue implementation ---

template <typename T>
class SafeQueue {
   private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;

   public:
    SafeQueue(size_t max_size = MAX_QUEUE_SIZE) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return queue_.size() < max_size_; });
        queue_.push(std::move(item));
        condition_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        condition_.notify_one();
        return item;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

// --- 3. Helper function define ---

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
void scale_coordinates(YOLOv11Result& detection, const std::vector<int>& pad_xy,
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
cv::Mat draw_detections(const cv::Mat& frame, std::vector<YOLOv11Result>& detections,
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

// --- 4. Thread function define ---

void post_process_thread_func(SafeQueue<std::shared_ptr<DetectionArgs>> *wait_queue,
                              SafeQueue<std::shared_ptr<DisplayArgs>> *display_queue,
                              std::atomic<int> *appQuit) {
    while (appQuit->load() == -1) std::this_thread::sleep_for(std::chrono::microseconds(10));

    while (appQuit->load() == 0) {
        if (wait_queue->empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }
        auto args = wait_queue->pop();

        auto outputs = args->ie->Wait(args->request_id);
        auto t1 = std::chrono::high_resolution_clock::now();
        double inference_time =
            std::chrono::duration<double, std::milli>(t1 - args->t_run_async_start).count();

        // Try postprocess timing
        // aligned tensor processing is now handled inside postprocess
        std::vector<YOLOv11Result> detections_vec;
        try {
            detections_vec = args->ypp->postprocess(outputs);
        } catch (const std::exception& e) {
            std::cerr << "[DXAPP] [ER] Exception during postprocessing: \n"
                        << e.what() << std::endl;
            appQuit->store(1);
            continue;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double postprocess_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (args->metrics) {
            std::lock_guard<std::mutex> lock(args->metrics->metrics_mutex);
            args->metrics->infer_last_ts = t1;
            args->metrics->infer_completed++;
            // Accumulate inflight time before decrementing
            auto now = std::chrono::high_resolution_clock::now();
            args->metrics->inflight_time_sum +=
                args->metrics->inflight_current *
                std::chrono::duration<double>(now - args->metrics->inflight_last_ts).count();
            args->metrics->inflight_last_ts = now;
            args->metrics->inflight_current--;
        }

        auto d_args = std::make_shared<DisplayArgs>();
        d_args->detections = std::make_shared<std::vector<YOLOv11Result>>(detections_vec);
        if (!args->current_frame.empty()) {
            d_args->original_frame = std::make_shared<cv::Mat>(args->current_frame.clone());
        }
        d_args->ypp = args->ypp;
        d_args->processed_count = args->processed_count;
        d_args->is_no_show = args->is_no_show;
        d_args->is_video_save = args->is_video_save;
        d_args->t_preprocess = args->t_preprocess;
        d_args->t_inference = inference_time;
        d_args->t_postprocess = postprocess_time;
        d_args->metrics = args->metrics;

        display_queue->push(d_args);
    }
}

void display_thread_func(SafeQueue<std::shared_ptr<DisplayArgs>> *display_queue,
                         std::atomic<int> *appQuit, cv::VideoWriter *writer,
                         std::vector<int> *pad_xy, float *scale_factor) {
    while (appQuit->load() == -1) std::this_thread::sleep_for(std::chrono::microseconds(10));

    while (appQuit->load() == 0 || !display_queue->empty()) {
        if (display_queue->empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto args = display_queue->pop();
        if (!args || !args->original_frame) continue;

        auto render_start = std::chrono::high_resolution_clock::now();
        auto processed_frame =
            draw_detections(*args->original_frame, *args->detections, *pad_xy, *scale_factor);

        if (!processed_frame.empty()) {
            if (args->is_video_save) *writer << processed_frame;
            if (!args->is_no_show) {
                cv::imshow("result", processed_frame);
                if (cv::waitKey(1) == 'q') appQuit->store(1);
            }
        }

        if (args->processed_count) (*args->processed_count)++;

        auto render_end = std::chrono::high_resolution_clock::now();
        double render_time =
            std::chrono::duration<double, std::milli>(render_end - render_start).count();

        if (args->metrics) {
            std::lock_guard<std::mutex> lock(args->metrics->metrics_mutex);
            args->metrics->sum_preprocess += args->t_preprocess;
            args->metrics->sum_inference += args->t_inference;
            args->metrics->sum_postprocess += args->t_postprocess;
            args->metrics->sum_render += render_time;
        }
    }
}

void print_performance_summary(const ProfilingMetrics &metrics, int total_frames,
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
    int loopTest = 1, processCount = 0;

    std::string app_name = "YOLOv11 Post-Processing Async Example";
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
        YOLOv11PostProcess(input_width, input_height, 0.3f, 0.45f, ie.IsOrtConfigured());

    // Print model input size
    std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height
              << std::endl;

    std::vector<std::vector<uint8_t>> input_buffers(ASYNC_BUFFER_SIZE,
                                                    std::vector<uint8_t>(ie.GetInputSize()));

    SafeQueue<std::shared_ptr<DetectionArgs>> wait_queue;
    SafeQueue<std::shared_ptr<DisplayArgs>> display_queue;
    ProfilingMetrics profiling_metrics;

    cv::VideoWriter writer;

    // Calculate letterbox padding and scale factor
    float scale_factor =
        std::min(static_cast<float>(input_width) / static_cast<float>(SHOW_WINDOW_SIZE_W),
                 static_cast<float>(input_height) / static_cast<float>(SHOW_WINDOW_SIZE_H));
    int letterbox_pad_x = static_cast<int>(
        std::max(0.f, (static_cast<float>(input_width) - SHOW_WINDOW_SIZE_W * scale_factor) / 2.f));
    int letterbox_pad_y = static_cast<int>(
        std::max(0.f, (static_cast<float>(input_height) - SHOW_WINDOW_SIZE_H * scale_factor) / 2.f));
    std::vector<int> pad_xy{letterbox_pad_x, letterbox_pad_y};

    std::thread post_thread(post_process_thread_func, &wait_queue, &display_queue, &appQuit);
    std::thread disp_thread(display_thread_func, &display_queue, &appQuit, &writer, &pad_xy,
                            &scale_factor);

    cv::VideoCapture video;
    bool is_image = !imgFile.empty();
    if (is_image) { /* Image logic below */
    } else if (cameraIndex >= 0)
        video.open(cameraIndex);
    else if (!rtspUrl.empty())
        video.open(rtspUrl);
    else
        video.open(videoFile);

    if (!is_image && !video.isOpened()) return -1;

    // Video Save Setup
    if (saveVideo) {
        double fps = video.get(cv::CAP_PROP_FPS);
        writer.open("result.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps > 0 ? fps : 30.0,
                    cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
    }

    std::vector<cv::Mat> images(ASYNC_BUFFER_SIZE,
                                cv::Mat(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3));
    int index = 0, submitted_frames = 0;
    auto s_time = std::chrono::high_resolution_clock::now();

    if (is_image) {
        cv::Mat img = cv::imread(imgFile);
        for (int i = 0; i < loopTest; ++i) {
            // Backpressure: wait if too many requests are in flight
            while (appQuit.load() <= 0) {
                {
                    std::lock_guard<std::mutex> lk(profiling_metrics.metrics_mutex);
                    if (profiling_metrics.inflight_current < ASYNC_BUFFER_SIZE - 1) break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (appQuit.load() > 0) break;

            auto t0 = std::chrono::high_resolution_clock::now();
            cv::resize(img, images[index], cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            cv::Mat pre(input_height, input_width, CV_8UC3, input_buffers[index].data());
            make_letterbox_image(images[index], pre, cv::COLOR_BGR2RGB, pad_xy);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto req_id = ie.RunAsync(pre.data, nullptr, nullptr);

            auto args = std::make_shared<DetectionArgs>();
            args->ie = &ie;
            args->ypp = &post_processor;
            args->current_frame = images[index].clone();
            args->request_id = req_id;
            args->processed_count = &processCount;
            args->metrics = &profiling_metrics;
            args->t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
            args->t_run_async_start = t1;
            args->is_no_show = fps_only;

            {
                std::lock_guard<std::mutex> lk(profiling_metrics.metrics_mutex);
                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1;
                    profiling_metrics.inflight_last_ts = t1;
                    profiling_metrics.first_inference = false;
                } else {
                    // Accumulate inflight time before incrementing
                    auto now = std::chrono::high_resolution_clock::now();
                    profiling_metrics.inflight_time_sum +=
                        profiling_metrics.inflight_current *
                        std::chrono::duration<double>(now - profiling_metrics.inflight_last_ts).count();
                    profiling_metrics.inflight_last_ts = now;
                }
                profiling_metrics.inflight_current++;
                if (profiling_metrics.inflight_current > profiling_metrics.inflight_max)
                    profiling_metrics.inflight_max = profiling_metrics.inflight_current;
            }

            wait_queue.push(args);
            submitted_frames++;
            if (appQuit.load() == -1) appQuit.store(0);
            index = (index + 1) % ASYNC_BUFFER_SIZE;
        }
    } else {
        while (true) {
            cv::Mat frame;
            auto tr0 = std::chrono::high_resolution_clock::now();
            video >> frame;
            if (frame.empty()) break;
            auto tr1 = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lk(profiling_metrics.metrics_mutex);
                profiling_metrics.sum_read +=
                    std::chrono::duration<double, std::milli>(tr1 - tr0).count();
            }

            // Backpressure: wait if too many requests are in flight
            while (appQuit.load() <= 0) {
                {
                    std::lock_guard<std::mutex> lk(profiling_metrics.metrics_mutex);
                    if (profiling_metrics.inflight_current < ASYNC_BUFFER_SIZE - 1) break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (appQuit.load() > 0) break;

            auto t0 = std::chrono::high_resolution_clock::now();
            cv::resize(frame, images[index], cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            cv::Mat pre(input_height, input_width, CV_8UC3, input_buffers[index].data());
            make_letterbox_image(images[index], pre, cv::COLOR_BGR2RGB, pad_xy);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto req_id = ie.RunAsync(pre.data, nullptr, nullptr);

            auto args = std::make_shared<DetectionArgs>();
            args->ie = &ie;
            args->ypp = &post_processor;
            args->current_frame = images[index].clone();
            args->request_id = req_id;
            args->processed_count = &processCount;
            args->metrics = &profiling_metrics;
            args->t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
            args->t_run_async_start = t1;
            args->is_no_show = fps_only;
            args->is_video_save = saveVideo;

            {
                std::lock_guard<std::mutex> lk(profiling_metrics.metrics_mutex);
                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1;
                    profiling_metrics.inflight_last_ts = t1;
                    profiling_metrics.first_inference = false;
                } else {
                    // Accumulate inflight time before incrementing
                    auto now = std::chrono::high_resolution_clock::now();
                    profiling_metrics.inflight_time_sum +=
                        profiling_metrics.inflight_current *
                        std::chrono::duration<double>(now - profiling_metrics.inflight_last_ts).count();
                    profiling_metrics.inflight_last_ts = now;
                }
                profiling_metrics.inflight_current++;
                if (profiling_metrics.inflight_current > profiling_metrics.inflight_max)
                    profiling_metrics.inflight_max = profiling_metrics.inflight_current;
            }

            wait_queue.push(args);
            submitted_frames++;
            if (appQuit.load() == -1) appQuit.store(0);
            index = (index + 1) % ASYNC_BUFFER_SIZE;
            if (appQuit.load() == 1) break;
        }
    }

    while (processCount < submitted_frames && appQuit.load() <= 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    appQuit.store(1);
    if (post_thread.joinable()) post_thread.join();
    if (disp_thread.joinable()) disp_thread.join();

    auto e_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(e_time - s_time).count();
    print_performance_summary(profiling_metrics, processCount, total_time, !fps_only);

    DXRT_TRY_CATCH_END
    return 0;
}
