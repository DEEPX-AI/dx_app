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

#include "yolov8seg_postprocess.h"

/**
 * @brief Asynchronous post-processing example for yolov8-seg instance segmentation model.
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

// Pre-computed color table for class visualization (optimized for performance)
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

struct DisplayArgs {
    std::shared_ptr<std::vector<YOLOv8SegResult>> detections;
    std::shared_ptr<cv::Mat> original_frame;

    YOLOv8SegPostProcess *ypp = nullptr;
    int *processed_count = nullptr;
    bool is_no_show = false;
    bool is_video_save = false;
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    ProfilingMetrics *metrics = nullptr;
    std::vector<int> pad_xy{0, 0};
    std::vector<float> ratio{1.0f, 1.0f};

    DisplayArgs() = default;
};

struct DetectionArgs {
    cv::Mat current_frame;
    dxrt::InferenceEngine *ie = nullptr;
    YOLOv8SegPostProcess *ypp = nullptr;
    ProfilingMetrics *metrics = nullptr;
    int *processed_count = nullptr;
    int request_id = 0;
    bool is_no_show = false;
    bool is_video_save = false;
    double t_preprocess = 0.0;
    std::chrono::high_resolution_clock::time_point t_run_async_start;
    std::vector<int> pad_xy{0, 0};
    std::vector<float> ratio{1.0f, 1.0f};

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
cv::Mat draw_yolov8_segmentation(const cv::Mat& frame, std::vector<YOLOv8SegResult>& results,
                                 const std::vector<int>& pad_xy, const std::vector<float>& ratio,
                                 const float alpha = 0.6f) {
    cv::Mat result = frame.clone();

    for (size_t i = 0; i < results.size(); ++i) {
        auto& detection = results[i];

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
                    for (int j = 0; j < 4; ++j) {
                        if (needs_blend[j]) {
                            cv::Vec3b& pixel = result_row[x + j];
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
        std::vector<YOLOv8SegResult> detections_vec;
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
        d_args->detections = std::make_shared<std::vector<YOLOv8SegResult>>(std::move(detections_vec));
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
        d_args->pad_xy = args->pad_xy;
        d_args->ratio = args->ratio;

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
            draw_yolov8_segmentation(*args->original_frame, *args->detections, args->pad_xy, args->ratio);

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

    std::string app_name = "YOLOv8-Seg Post-Processing Async Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "instance segmentation model file (.dxnn, required)",
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
        YOLOv8SegPostProcess(input_width, input_height, 0.3f, 0.45f, ie.IsOrtConfigured());

    // Print model input size
    std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height
              << std::endl;

    std::vector<std::vector<uint8_t>> input_buffers(ASYNC_BUFFER_SIZE,
                                                    std::vector<uint8_t>(ie.GetInputSize()));

    SafeQueue<std::shared_ptr<DetectionArgs>> wait_queue;
    SafeQueue<std::shared_ptr<DisplayArgs>> display_queue;
    ProfilingMetrics profiling_metrics;

    cv::VideoWriter writer;

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

            std::vector<int> pad_xy{0, 0};
            std::vector<float> ratio{1.0f, 1.0f};

            auto t0 = std::chrono::high_resolution_clock::now();
            cv::resize(img, images[index], cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            cv::Mat pre(input_height, input_width, CV_8UC3, input_buffers[index].data());
            make_letterbox_image(images[index], pre, pad_xy, ratio);
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
            args->pad_xy = pad_xy;
            args->ratio = ratio;

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

            std::vector<int> pad_xy{0, 0};
            std::vector<float> ratio{1.0f, 1.0f};

            auto t0 = std::chrono::high_resolution_clock::now();
            cv::resize(frame, images[index], cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            cv::Mat pre(input_height, input_width, CV_8UC3, input_buffers[index].data());
            make_letterbox_image(images[index], pre, pad_xy, ratio);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto req_id = ie.RunAsync(pre.data, nullptr, nullptr);
            auto t2 = std::chrono::high_resolution_clock::now();

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
            args->pad_xy = pad_xy;
            args->ratio = ratio;

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
