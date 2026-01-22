#include <deeplabv3_postprocess.h>
#include <dxrt/dxrt_api.h>
#include <yolov7_postprocess.h>

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

/**
 * @brief Asynchronous multi-model example combining YOLOv7 object detection and
 * DeepLabV3 semantic segmentation.
 *
 * - Supports image, video, and camera input sources.
 * - Runs both YOLOv7 and DeepLabV3 models asynchronously on the same input.
 * - Performs post-processing on both model inference results simultaneously.
 * - Combines visualization of object detection boxes and semantic segmentation masks.
 * - Command-line options allow configuration of model paths, input files, loop
 * count, FPS measurement, and result saving.
 *
 * Variable declarations and main logic are written for maintainability and code
 * optimization with dual-model async processing.
 */

#define ASYNC_BUFFER_SIZE 40
#define MAX_QUEUE_SIZE 100

#define SHOW_WINDOW_SIZE_W 960
#define SHOW_WINDOW_SIZE_H 640

// Combined results structure
struct CombinedResults {
    std::vector<YOLOv7Result> detections;
    DeepLabv3Result segmentation;

    CombinedResults() {}

    CombinedResults(const std::vector<YOLOv7Result>& det, const DeepLabv3Result& seg)
        : detections(det), segmentation(seg) {}

    CombinedResults(std::vector<YOLOv7Result>&& det, DeepLabv3Result&& seg)
        : detections(std::move(det)), segmentation(std::move(seg)) {}
};

// Profiling metrics structure for dual models
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
    std::shared_ptr<CombinedResults> combined_results;
    std::shared_ptr<cv::Mat> original_frame;

    YOLOv7PostProcess *ypp = nullptr;
    DeepLabv3PostProcess *dlpp = nullptr;
    int *processed_count = nullptr;
    bool is_no_show = false;
    bool is_video_save = false;

    // Timing information
    double t_yolo_preprocess = 0.0;
    double t_deeplab_preprocess = 0.0;
    double t_yolo_inference = 0.0;
    double t_deeplab_inference = 0.0;
    double t_yolo_postprocess = 0.0;
    double t_deeplab_postprocess = 0.0;
    ProfilingMetrics *metrics = nullptr;

    DisplayArgs() = default;
};

struct DetectionArgs {
    cv::Mat current_frame;
    dxrt::InferenceEngine *yolo_ie = nullptr;
    dxrt::InferenceEngine *deeplab_ie = nullptr;
    YOLOv7PostProcess *ypp = nullptr;
    DeepLabv3PostProcess *dlpp = nullptr;
    ProfilingMetrics *metrics = nullptr;
    int *processed_count = nullptr;
    int yolo_request_id = 0;
    int deeplab_request_id = 0;
    bool is_no_show = false;
    bool is_video_save = false;

    // Timing information
    double t_yolo_preprocess = 0.0;
    double t_deeplab_preprocess = 0.0;
    std::chrono::high_resolution_clock::time_point t_yolo_async_start;
    std::chrono::high_resolution_clock::time_point t_deeplab_async_start;

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

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        condition_.notify_one();
        return true;
    }

    bool try_push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) return false;
        queue_.push(std::move(item));
        condition_.notify_one();
        return true;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void notify_all() {
        condition_.notify_all();
    }
};

// --- 3. Helper function define ---

// Generate color for each detection class ID
cv::Scalar get_coco_class_color(int class_id) {
    std::srand(class_id + 100);  // Different seed than segmentation
    int b = std::rand() % 256;
    int g = std::rand() % 256;
    int r = std::rand() % 256;
    return cv::Scalar(b, g, r);
}

// Generate color for each segmentation class ID (Cityscapes palette)
cv::Scalar get_cityscapes_class_color(int class_id) {
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
    return cv::Scalar(0, 0, 0);  // Black for unknown classes
}

/**
 * @brief Resize the input image to the specified size and apply letterbox padding
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
 * @brief Scale detection coordinates from letterbox back to original coordinates
 */
void scale_coordinates(YOLOv7Result& detection, const std::vector<int>& pad_xy,
                       const float letterbox_scale) {
    detection.box[0] = (detection.box[0] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[1] = (detection.box[1] - static_cast<float>(pad_xy[1])) / letterbox_scale;
    detection.box[2] = (detection.box[2] - static_cast<float>(pad_xy[0])) / letterbox_scale;
    detection.box[3] = (detection.box[3] - static_cast<float>(pad_xy[1])) / letterbox_scale;
}

/**
 * @brief Scale segmentation mask from letterbox back to original size
 */
cv::Mat scale_segmentation_mask(const cv::Mat& mask, int orig_width, int orig_height,
                                const std::vector<int>& pad_xy) {
    int unpad_w = mask.cols - 2 * pad_xy[0];
    int unpad_h = mask.rows - 2 * pad_xy[1];

    cv::Mat unpadded_mask;
    if (pad_xy[0] > 0 || pad_xy[1] > 0) {
        cv::Rect crop_region(pad_xy[0], pad_xy[1], unpad_w, unpad_h);
        unpadded_mask = mask(crop_region).clone();
    } else {
        unpadded_mask = mask.clone();
    }

    cv::Mat resized_mask;
    cv::resize(unpadded_mask, resized_mask, cv::Size(orig_width, orig_height), 0, 0,
               cv::INTER_NEAREST);

    return resized_mask;
}

/**
 * @brief Visualize combined results - both detection boxes and segmentation masks
 */
cv::Mat draw_combined_results(const cv::Mat& frame, CombinedResults& results,
                              const std::vector<int>& pad_xy, const float letterbox_scale,
                              const float seg_alpha = 0.4f) {
    static std::vector<int> pad_xy_static{0, 0};
    cv::Mat result = frame.clone();

    // First, draw segmentation mask as background overlay
    if (!results.segmentation.segmentation_mask.empty() && results.segmentation.width > 0 &&
        results.segmentation.height > 0) {
        // Create mask image from segmentation result
        cv::Mat mask_image =
            cv::Mat::zeros(results.segmentation.height, results.segmentation.width, CV_8UC3);

        // Fill mask with class colors
        for (int y = 0; y < results.segmentation.height; ++y) {
            for (int x = 0; x < results.segmentation.width; ++x) {
                int idx = y * results.segmentation.width + x;
                if (idx < static_cast<int>(results.segmentation.segmentation_mask.size())) {
                    int class_id = results.segmentation.segmentation_mask[idx];
                    cv::Scalar color = get_cityscapes_class_color(class_id);
                    mask_image.at<cv::Vec3b>(y, x) =
                        cv::Vec3b(static_cast<uchar>(color[0]), static_cast<uchar>(color[1]),
                                  static_cast<uchar>(color[2]));
                }
            }
        }

        // Scale mask to match original frame size
        cv::Mat scaled_mask = scale_segmentation_mask(mask_image, frame.cols, frame.rows, pad_xy_static);

        // Blend the mask with original image
        cv::addWeighted(result, 1.0 - seg_alpha, scaled_mask, seg_alpha, 0, result);
    }

    // Then, draw detection boxes on top
    for (auto& detection : results.detections) {
        scale_coordinates(detection, pad_xy, letterbox_scale);

        // Get class-specific color for detection
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
        cv::rectangle(result, bg_tl, bg_br, cv::Scalar(0, 0, 0), -1);

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

    while (appQuit->load() == 0 || !wait_queue->empty()) {
        std::shared_ptr<DetectionArgs> args;
        if (!wait_queue->try_pop(args)) {
            if (appQuit->load() != 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        // Wait for both model outputs
        auto yolo_outputs = args->yolo_ie->Wait(args->yolo_request_id);
        auto t_yolo_end = std::chrono::high_resolution_clock::now();

        auto deeplab_outputs = args->deeplab_ie->Wait(args->deeplab_request_id);
        auto t_deeplab_end = std::chrono::high_resolution_clock::now();

        double yolo_inference_time =
            std::chrono::duration<double, std::milli>(t_yolo_end - args->t_yolo_async_start).count();
        double deeplab_inference_time =
            std::chrono::duration<double, std::milli>(t_deeplab_end - args->t_deeplab_async_start).count();

        // Postprocess YOLOv7
        std::vector<YOLOv7Result> detection_results;
        try {
            detection_results = args->ypp->postprocess(yolo_outputs);
        } catch (const std::exception& e) {
            std::cerr << "[DXAPP] [ER] Exception during YOLOv7 postprocessing: \n"
                      << e.what() << std::endl;
            appQuit->store(1);
            continue;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double yolo_postprocess_time =
            std::chrono::duration<double, std::milli>(t2 - t_yolo_end).count();

        // Postprocess DeepLabV3
        DeepLabv3Result segmentation_result;
        try {
            segmentation_result = args->dlpp->postprocess(deeplab_outputs);
        } catch (const std::exception& e) {
            std::cerr << "[DXAPP] [ER] Exception during DeepLabV3 postprocessing: \n"
                      << e.what() << std::endl;
            appQuit->store(1);
            continue;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double deeplab_postprocess_time =
            std::chrono::duration<double, std::milli>(t3 - t2).count();

        if (args->metrics) {
            std::lock_guard<std::mutex> lock(args->metrics->metrics_mutex);
            args->metrics->infer_last_ts = std::max(t_yolo_end, t_deeplab_end);
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
        d_args->combined_results = std::make_shared<CombinedResults>(
            std::move(detection_results), std::move(segmentation_result));
        if (!args->current_frame.empty()) {
            d_args->original_frame = std::make_shared<cv::Mat>(args->current_frame.clone());
        }
        d_args->ypp = args->ypp;
        d_args->dlpp = args->dlpp;
        d_args->processed_count = args->processed_count;
        d_args->is_no_show = args->is_no_show;
        d_args->is_video_save = args->is_video_save;
        d_args->t_yolo_preprocess = args->t_yolo_preprocess;
        d_args->t_deeplab_preprocess = args->t_deeplab_preprocess;
        d_args->t_yolo_inference = yolo_inference_time;
        d_args->t_deeplab_inference = deeplab_inference_time;
        d_args->t_yolo_postprocess = yolo_postprocess_time;
        d_args->t_deeplab_postprocess = deeplab_postprocess_time;
        d_args->metrics = args->metrics;

        while (!display_queue->try_push(d_args)) {
            if (appQuit->load() != 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void display_thread_func(SafeQueue<std::shared_ptr<DisplayArgs>> *display_queue,
                         std::atomic<int> *appQuit, cv::VideoWriter *writer,
                         std::vector<int> *pad_xy, float *scale_factor) {
    while (appQuit->load() == -1) std::this_thread::sleep_for(std::chrono::microseconds(10));

    while (appQuit->load() == 0 || !display_queue->empty()) {
        std::shared_ptr<DisplayArgs> args;
        if (!display_queue->try_pop(args)) {
            if (appQuit->load() != 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (!args || !args->original_frame) continue;

        auto render_start = std::chrono::high_resolution_clock::now();
        auto processed_frame =
            draw_combined_results(*args->original_frame, *args->combined_results, *pad_xy, *scale_factor);

        if (!processed_frame.empty()) {
            if (args->is_video_save) *writer << processed_frame;
            if (!args->is_no_show) {
                cv::imshow("YOLOv7 + DeepLabV3 Combined Result", processed_frame);
                if (cv::waitKey(1) == 'q') appQuit->store(1);
            }
        }

        if (args->processed_count) (*args->processed_count)++;

        auto render_end = std::chrono::high_resolution_clock::now();
        double render_time =
            std::chrono::duration<double, std::milli>(render_end - render_start).count();

        if (args->metrics) {
            std::lock_guard<std::mutex> lock(args->metrics->metrics_mutex);
            args->metrics->sum_yolo_preprocess += args->t_yolo_preprocess;
            args->metrics->sum_deeplab_preprocess += args->t_deeplab_preprocess;
            args->metrics->sum_yolo_inference += args->t_yolo_inference;
            args->metrics->sum_deeplab_inference += args->t_deeplab_inference;
            args->metrics->sum_yolo_postprocess += args->t_yolo_postprocess;
            args->metrics->sum_deeplab_postprocess += args->t_deeplab_postprocess;
            args->metrics->sum_render += render_time;
        }
    }
}

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec, bool display_on) {
    if (metrics.infer_completed == 0) return;

    auto safe_avg = [&](double sum) {
        return (metrics.infer_completed > 0) ? sum / metrics.infer_completed : 0.0;
    };

    double avg_read = safe_avg(metrics.sum_read);
    double avg_yolo_pre = safe_avg(metrics.sum_yolo_preprocess);
    double avg_yolo_inf = safe_avg(metrics.sum_yolo_inference);
    double avg_yolo_post = safe_avg(metrics.sum_yolo_postprocess);
    double avg_deeplab_pre = safe_avg(metrics.sum_deeplab_preprocess);
    double avg_deeplab_inf = safe_avg(metrics.sum_deeplab_inference);
    double avg_deeplab_post = safe_avg(metrics.sum_deeplab_postprocess);

    auto inflight_time_window =
        std::chrono::duration<double>(metrics.infer_last_ts - metrics.infer_first_ts).count();

    double infer_tp =
        (inflight_time_window > 0) ? metrics.infer_completed / inflight_time_window : 0.0;
    double inflight_avg =
        (inflight_time_window > 0) ? metrics.inflight_time_sum / inflight_time_window : 0.0;

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
        double avg_render = metrics.sum_render / metrics.infer_completed;
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
    std::cout << " * Actual throughput via async inference" << std::endl;
    std::cout << "   (shared across YOLOv7 and DeepLabV3 pipelines)" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Async Throughput"
              << " :    " << std::fixed << std::setprecision(1) << infer_tp << " FPS"
              << std::endl;
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
    std::string yoloModelPath = "", deeplabModelPath = "", imgFile = "", videoFile = "", rtspUrl = "";
    int cameraIndex = -1;
    bool fps_only = false, saveVideo = false;
    int loopTest = 1, processCount = 0;

    std::string app_name = "YOLOv7 + DeepLabV3 Multi-Model Async Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("y, yolo_model", "YOLOv7 object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(yoloModelPath))(
        "d, deeplab_model", "DeepLabV3 segmentation model file (.dxnn, required)",
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
        std::cerr << "[ERROR] DeepLabV3 model path is required. Use -d or "
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

    // Initialize both inference engines
    dxrt::InferenceOption io;
    dxrt::InferenceEngine yolo_ie(yoloModelPath, io);
    dxrt::InferenceEngine deeplab_ie(deeplabModelPath, io);

    if (!dxapp::common::minversionforRTandCompiler(&yolo_ie) ||
        !dxapp::common::minversionforRTandCompiler(&deeplab_ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the "
                     "version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    // Get model input dimensions
    auto yolo_input_shape = yolo_ie.GetInputs().front().shape();
    auto deeplab_input_shape = deeplab_ie.GetInputs().front().shape();

    int yolo_input_height = static_cast<int>(yolo_input_shape[1]);
    int yolo_input_width = static_cast<int>(yolo_input_shape[2]);
    int deeplab_input_height = static_cast<int>(deeplab_input_shape[1]);
    int deeplab_input_width = static_cast<int>(deeplab_input_shape[2]);

    auto yolo_post_processor = YOLOv7PostProcess(yolo_input_width, yolo_input_height, 0.25f, 0.25f,
                                                 0.45f, yolo_ie.IsOrtConfigured());
    auto deeplab_post_processor = DeepLabv3PostProcess(deeplab_input_width, deeplab_input_height);

    // Print model information
    std::cout << "[INFO] YOLOv7 Model input size (WxH): " << yolo_input_width << "x"
              << yolo_input_height << std::endl;
    std::cout << "[INFO] DeepLabV3 Model input size (WxH): " << deeplab_input_width << "x"
              << deeplab_input_height << std::endl;

    // Prepare async buffers for both models
    std::vector<std::vector<uint8_t>> yolo_input_buffers(ASYNC_BUFFER_SIZE,
                                                         std::vector<uint8_t>(yolo_ie.GetInputSize()));
    std::vector<std::vector<uint8_t>> deeplab_input_buffers(ASYNC_BUFFER_SIZE,
                                                            std::vector<uint8_t>(deeplab_ie.GetInputSize()));

    SafeQueue<std::shared_ptr<DetectionArgs>> wait_queue;
    SafeQueue<std::shared_ptr<DisplayArgs>> display_queue;
    ProfilingMetrics profiling_metrics;

    cv::VideoWriter writer;

    // Calculate letterbox padding and scale factor
    float yolo_scale_factor =
        std::min(static_cast<float>(yolo_input_width) / static_cast<float>(SHOW_WINDOW_SIZE_W),
                 static_cast<float>(yolo_input_height) / static_cast<float>(SHOW_WINDOW_SIZE_H));
    int letterbox_pad_x = static_cast<int>(
        std::max(0.f, (static_cast<float>(yolo_input_width) - SHOW_WINDOW_SIZE_W * yolo_scale_factor) / 2.f));
    int letterbox_pad_y = static_cast<int>(
        std::max(0.f, (static_cast<float>(yolo_input_height) - SHOW_WINDOW_SIZE_H * yolo_scale_factor) / 2.f));
    std::vector<int> yolo_pad_xy{letterbox_pad_x, letterbox_pad_y};
    std::vector<int> deeplab_pad_xy{0, 0};

    std::thread post_thread(post_process_thread_func, &wait_queue, &display_queue, &appQuit);
    std::thread disp_thread(display_thread_func, &display_queue, &appQuit, &writer, &yolo_pad_xy,
                            &yolo_scale_factor);

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

            // Preprocess for YOLOv7
            cv::Mat yolo_pre(yolo_input_height, yolo_input_width, CV_8UC3,
                             yolo_input_buffers[index].data());
            make_letterbox_image(images[index], yolo_pre, cv::COLOR_BGR2RGB, yolo_pad_xy);
            auto t1_yolo = std::chrono::high_resolution_clock::now();

            // Preprocess for DeepLabV3
            cv::Mat deeplab_pre(deeplab_input_height, deeplab_input_width, CV_8UC3,
                                deeplab_input_buffers[index].data());
            make_letterbox_image(images[index], deeplab_pre, cv::COLOR_BGR2RGB, deeplab_pad_xy);
            auto t1_deeplab = std::chrono::high_resolution_clock::now();

            double yolo_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_yolo - t0).count();
            double deeplab_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_deeplab - t1_yolo).count();

            auto yolo_req_id = yolo_ie.RunAsync(yolo_pre.data, nullptr, nullptr);
            auto deeplab_req_id = deeplab_ie.RunAsync(deeplab_pre.data, nullptr, nullptr);

            auto args = std::make_shared<DetectionArgs>();
            args->yolo_ie = &yolo_ie;
            args->deeplab_ie = &deeplab_ie;
            args->ypp = &yolo_post_processor;
            args->dlpp = &deeplab_post_processor;
            args->current_frame = images[index].clone();
            args->yolo_request_id = yolo_req_id;
            args->deeplab_request_id = deeplab_req_id;
            args->processed_count = &processCount;
            args->metrics = &profiling_metrics;
            args->t_yolo_preprocess = yolo_preprocess_time;
            args->t_deeplab_preprocess = deeplab_preprocess_time;
            args->t_yolo_async_start = t1_yolo;
            args->t_deeplab_async_start = t1_deeplab;
            args->is_no_show = fps_only;

            {
                std::lock_guard<std::mutex> lk(profiling_metrics.metrics_mutex);
                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1_yolo;
                    profiling_metrics.inflight_last_ts = t1_yolo;
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

            while (!wait_queue.try_push(args)) {
                if (appQuit.load() == 1) break;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            if (appQuit.load() == 1) break;
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

            // Preprocess for YOLOv7
            cv::Mat yolo_pre(yolo_input_height, yolo_input_width, CV_8UC3,
                             yolo_input_buffers[index].data());
            make_letterbox_image(images[index], yolo_pre, cv::COLOR_BGR2RGB, yolo_pad_xy);
            auto t1_yolo = std::chrono::high_resolution_clock::now();

            // Preprocess for DeepLabV3
            cv::Mat deeplab_pre(deeplab_input_height, deeplab_input_width, CV_8UC3,
                                deeplab_input_buffers[index].data());
            make_letterbox_image(images[index], deeplab_pre, cv::COLOR_BGR2RGB, deeplab_pad_xy);
            auto t1_deeplab = std::chrono::high_resolution_clock::now();

            double yolo_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_yolo - t0).count();
            double deeplab_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_deeplab - t1_yolo).count();

            auto yolo_req_id = yolo_ie.RunAsync(yolo_pre.data, nullptr, nullptr);
            auto deeplab_req_id = deeplab_ie.RunAsync(deeplab_pre.data, nullptr, nullptr);
            auto t2 = std::chrono::high_resolution_clock::now();

            auto args = std::make_shared<DetectionArgs>();
            args->yolo_ie = &yolo_ie;
            args->deeplab_ie = &deeplab_ie;
            args->ypp = &yolo_post_processor;
            args->dlpp = &deeplab_post_processor;
            args->current_frame = images[index].clone();
            args->yolo_request_id = yolo_req_id;
            args->deeplab_request_id = deeplab_req_id;
            args->processed_count = &processCount;
            args->metrics = &profiling_metrics;
            args->t_yolo_preprocess = yolo_preprocess_time;
            args->t_deeplab_preprocess = deeplab_preprocess_time;
            args->t_yolo_async_start = t1_yolo;
            args->t_deeplab_async_start = t1_deeplab;
            args->is_no_show = fps_only;
            args->is_video_save = saveVideo;

            {
                std::lock_guard<std::mutex> lk(profiling_metrics.metrics_mutex);
                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1_yolo;
                    profiling_metrics.inflight_last_ts = t1_yolo;
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

            while (!wait_queue.try_push(args)) {
                if (appQuit.load() == 1) break;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            if (appQuit.load() == 1) break;
            submitted_frames++;
            if (appQuit.load() == -1) appQuit.store(0);
            index = (index + 1) % ASYNC_BUFFER_SIZE;
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

    std::cout << "\nMulti-model example completed successfully!" << std::endl;
    DXRT_TRY_CATCH_END
    return 0;
}
