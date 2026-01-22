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

#include "deeplabv3_postprocess.h"

/**
 * @brief Asynchronous post-processing example for DeepLabv3 semantic segmentation model.
 *
 * - Supports image, video, and camera input sources.
 * - Performs post-processing on model inference results (argmax, class prediction,
 *   semantic segmentation mask generation, etc.).
 * - Visualization and result saving are available using OpenCV.
 * - Command-line options allow configuration of model path, input files, loop
 * count, FPS measurement, and result saving.
 *
 * Variable declarations and main logic are written for maintainability and code
 * optimization.
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

// Generate color for each class ID using predefined colors for segmentation
cv::Scalar get_class_color(int class_id) {
    // Use predefined colors for better visualization of semantic classes
    // Cityscapes color palette for urban scene segmentation
    static const std::vector<cv::Scalar> colors = {
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

struct DisplayArgs {
    std::shared_ptr<DeepLabv3Result> segmentation_result;
    std::shared_ptr<cv::Mat> original_frame;

    DeepLabv3PostProcess *dlpp = nullptr;
    int *processed_count = nullptr;
    bool is_no_show = false;
    bool is_video_save = false;
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    ProfilingMetrics *metrics = nullptr;
    std::vector<int> pad_xy{0, 0};

    DisplayArgs() = default;
};

struct DetectionArgs {
    cv::Mat current_frame;
    dxrt::InferenceEngine *ie = nullptr;
    DeepLabv3PostProcess *dlpp = nullptr;
    ProfilingMetrics *metrics = nullptr;
    int *processed_count = nullptr;
    int request_id = 0;
    bool is_no_show = false;
    bool is_video_save = false;
    double t_preprocess = 0.0;
    std::chrono::high_resolution_clock::time_point t_run_async_start;
    std::vector<int> pad_xy{0, 0};

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

/**
 * @brief Resize the input image to the specified size and apply letterbox
 * padding for preprocessing semantic segmentation.
 * @param image Original input image
 * @param preprocessed_image Mat object to store the preprocessed result (already sized)
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
 * @brief Convert segmentation mask from letterbox padded/scaled coordinates back
 * to original image coordinates.
 * @param mask Segmentation mask to convert
 * @param orig_width Original image width
 * @param orig_height Original image height
 * @param pad_xy [x, y] vector for padding size
 * @return Resized segmentation mask
 */
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

/**
 * @brief Visualize segmentation results by overlaying colored mask on the image.
 * @param frame Original image
 * @param segmentation_result Segmentation result containing mask
 * @param pad_xy [x, y] vector for padding size
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
                cv::Scalar color = get_class_color(class_id);
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

        // Postprocess timing
        DeepLabv3Result segmentation_result;
        try {
            segmentation_result = args->dlpp->postprocess(outputs);
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
        d_args->segmentation_result =
            std::make_shared<DeepLabv3Result>(std::move(segmentation_result));
        if (!args->current_frame.empty()) {
            d_args->original_frame = std::make_shared<cv::Mat>(args->current_frame.clone());
        }
        d_args->dlpp = args->dlpp;
        d_args->processed_count = args->processed_count;
        d_args->is_no_show = args->is_no_show;
        d_args->is_video_save = args->is_video_save;
        d_args->t_preprocess = args->t_preprocess;
        d_args->t_inference = inference_time;
        d_args->t_postprocess = postprocess_time;
        d_args->metrics = args->metrics;
        d_args->pad_xy = args->pad_xy;

        display_queue->push(d_args);
    }
}

void display_thread_func(SafeQueue<std::shared_ptr<DisplayArgs>> *display_queue,
                         std::atomic<int> *appQuit, cv::VideoWriter *writer) {
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
            draw_segmentation(*args->original_frame, *args->segmentation_result, args->pad_xy);

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

    std::string app_name = "DeepLabv3 Post-Processing Async Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "semantic segmentation model file (.dxnn, required)",
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
    auto post_processor = DeepLabv3PostProcess(input_width, input_height);

    // Print model input size
    std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height
              << std::endl;

    std::vector<std::vector<uint8_t>> input_buffers(ASYNC_BUFFER_SIZE,
                                                    std::vector<uint8_t>(ie.GetInputSize()));

    SafeQueue<std::shared_ptr<DetectionArgs>> wait_queue;
    SafeQueue<std::shared_ptr<DisplayArgs>> display_queue;
    ProfilingMetrics profiling_metrics;

    cv::VideoWriter writer;
    std::vector<int> pad_xy = {0, 0};

    std::thread post_thread(post_process_thread_func, &wait_queue, &display_queue, &appQuit);
    std::thread disp_thread(display_thread_func, &display_queue, &appQuit, &writer);

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
            args->dlpp = &post_processor;
            args->current_frame = images[index].clone();
            args->request_id = req_id;
            args->processed_count = &processCount;
            args->metrics = &profiling_metrics;
            args->t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
            args->t_run_async_start = t1;
            args->is_no_show = fps_only;
            args->pad_xy = pad_xy;
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
            auto t2 = std::chrono::high_resolution_clock::now();

            auto args = std::make_shared<DetectionArgs>();
            args->ie = &ie;
            args->dlpp = &post_processor;
            args->current_frame = images[index].clone();
            args->request_id = req_id;
            args->processed_count = &processCount;
            args->metrics = &profiling_metrics;
            args->t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
            args->t_run_async_start = t1;
            args->is_no_show = fps_only;
            args->is_video_save = saveVideo;
            args->pad_xy = pad_xy;

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
