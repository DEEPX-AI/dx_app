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

#include "yolov7_postprocess.h"
#include "deeplabv3_postprocess.h"

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
 */


constexpr size_t ASYNC_BUFFER_SIZE = 40;
constexpr size_t MAX_QUEUE_SIZE = 100;

constexpr size_t SHOW_WINDOW_SIZE_W = 960;
constexpr size_t SHOW_WINDOW_SIZE_H = 640;

// --- Structures ---

// Combined results structure
struct CombinedResults {
    std::vector<YOLOv7Result> detections;
    DeepLabv3Result segmentation;

    CombinedResults() = default;
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
    double sum_deeplab_render = 0.0;
    double sum_yolo_render = 0.0;
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

// Command line arguments structure
struct CommandLineArgs {
    std::string yoloModelPath;
    std::string deeplabModelPath;
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
inline cv::Scalar get_coco_class_color(int class_id) {
    return COCO_CLASS_COLORS[class_id % COCO_CLASS_COLORS.size()];
}

// Pre-computed color table for semantic class visualization (Cityscapes palette)
static const std::vector<cv::Scalar> CITYSCAPES_COLORS = {
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

// Generate color for each class ID using pre-computed table
inline cv::Scalar get_cityscapes_class_color(int class_id) {
    return CITYSCAPES_COLORS[class_id % CITYSCAPES_COLORS.size()];
}

struct DisplayArgs {
    std::shared_ptr<CombinedResults> combined_results;
    std::shared_ptr<cv::Mat> original_frame;

    YOLOv7PostProcess *ypp = nullptr;
    DeepLabv3PostProcess *dlpp = nullptr;
    int *processed_count = nullptr;
    bool is_no_show = false;
    bool is_video_save = false;

    std::vector<int> pad_xy{0, 0};
    std::vector<float> ratio{1.0f, 1.0f};

    double t_read = 0.0;
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

    std::vector<int> pad_xy{0, 0};
    std::vector<float> ratio{1.0f, 1.0f};

    double t_read = 0.0;
    double t_yolo_preprocess = 0.0;
    double t_deeplab_preprocess = 0.0;
    std::chrono::high_resolution_clock::time_point t_yolo_async_start;
    std::chrono::high_resolution_clock::time_point t_deeplab_async_start;

    DetectionArgs() = default;
};

// --- SafeQueue implementation ---

template <typename T>
class SafeQueue {
   private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;

   public:
    explicit SafeQueue(size_t max_size = MAX_QUEUE_SIZE) : max_size_(max_size) {}

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

    // Try to pop with timeout, returns true if successful
    bool try_pop(T& item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!condition_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false;
        }
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
};

// --- Helper functions for postprocessing ---

bool handle_postprocess_exception(const std::exception& e, const std::string& context) {
    std::cerr << "[DXAPP] [ER] " << context << " error during postprocessing: \n"
              << e.what() << std::endl;
    return false;
}

bool process_yolo_postprocess(YOLOv7PostProcess& post_processor, const std::vector<std::shared_ptr<dxrt::Tensor>>& outputs, std::vector<YOLOv7Result>& detection_results) {
    try {
        detection_results = post_processor.postprocess(outputs);
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

bool process_deeplab_postprocess(DeepLabv3PostProcess& post_processor, const std::vector<std::shared_ptr<dxrt::Tensor>>& outputs, DeepLabv3Result& segmentation_result) {
    try {
        segmentation_result = post_processor.postprocess(outputs);
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

void update_inflight_metrics(ProfilingMetrics* metrics, const std::chrono::high_resolution_clock::time_point& t1) {
    std::lock_guard<std::mutex> lock(metrics->metrics_mutex);
    metrics->infer_last_ts = t1;
    metrics->infer_completed++;
    // Accumulate inflight time before decrementing
    auto now = std::chrono::high_resolution_clock::now();
    metrics->inflight_time_sum +=
        metrics->inflight_current *
        std::chrono::duration<double>(now - metrics->inflight_last_ts).count();
    metrics->inflight_last_ts = now;
    metrics->inflight_current--;
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

// Helper function to update profiling metrics inflight stats
void update_profiling_inflight(ProfilingMetrics& metrics, 
                               const std::chrono::high_resolution_clock::time_point& t1) {
    std::lock_guard<std::mutex> lk(metrics.metrics_mutex);
    if (metrics.first_inference) {
        metrics.infer_first_ts = t1;
        metrics.inflight_last_ts = t1;
        metrics.first_inference = false;
    }
    auto now = std::chrono::high_resolution_clock::now();
    metrics.inflight_time_sum +=
        metrics.inflight_current *
        std::chrono::duration<double>(now - metrics.inflight_last_ts).count();
    metrics.inflight_last_ts = now;

    metrics.inflight_current++;
    if (metrics.inflight_current > metrics.inflight_max)
        metrics.inflight_max = metrics.inflight_current;
}


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
 * @brief Draw a single detection bounding box with label on the image.
 * @param result Image to draw on (modified in place)
 * @param box Box coordinates [x1, y1, x2, y2]
 * @param detection Detection result containing class info
 * @param color Color for the bounding box
 */
void draw_detection_box(cv::Mat& result, const std::vector<float>& box,
                        const YOLOv7Result& detection, const cv::Scalar& color) {
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
cv::Mat draw_detections(const cv::Mat& frame, const std::vector<YOLOv7Result>& detections,
                        const std::vector<int>& pad_xy, const std::vector<float>& ratio) {
    cv::Mat result = frame.clone();

    for (const auto& detection : detections) {
        // Transform bounding box to original coordinates
        std::vector<float> box = detection.box;
        transform_box_to_original(box, pad_xy, ratio, frame.cols, frame.rows);

        // Get class-specific color
        cv::Scalar color = get_coco_class_color(detection.class_id);
        draw_detection_box(result, box, detection, color);
    }

    return result;
}


/**
 * @brief Create a colored segmentation mask from class IDs.
 */
cv::Mat create_segmentation_mask(const DeepLabv3Result& segmentation) {
    cv::Mat mask_image = cv::Mat::zeros(segmentation.height, segmentation.width, CV_8UC3);
    
    for (int y = 0; y < segmentation.height; ++y) {
        for (int x = 0; x < segmentation.width; ++x) {
            int idx = y * segmentation.width + x;
            if (idx < static_cast<int>(segmentation.segmentation_mask.size())) {
                int class_id = segmentation.segmentation_mask[idx];
                cv::Scalar color = get_cityscapes_class_color(class_id);
                mask_image.at<cv::Vec3b>(y, x) =
                    cv::Vec3b(static_cast<uchar>(color[0]), static_cast<uchar>(color[1]),
                              static_cast<uchar>(color[2]));
            }
        }
    }
    
    return mask_image;
}

/**
 * @brief Draw segmentation mask on the frame with alpha blending.
 * @param frame Original image
 * @param segmentation Segmentation result
 * @param pad_xy [x, y] vector for padding size (unused for segmentation)
 * @param seg_alpha Alpha value for blending (default 0.4)
 * @return Image with segmentation overlay
 */
cv::Mat draw_segmentation(const cv::Mat& frame, const DeepLabv3Result& segmentation,
                         const std::vector<int>& pad_xy, const float seg_alpha = 0.4f) {
    std::ignore = pad_xy;
    static std::vector<int> pad_xy_static{0, 0};
    cv::Mat result = frame.clone();

    if (!segmentation.segmentation_mask.empty() && segmentation.width > 0 &&
        segmentation.height > 0) {
        cv::Mat mask_image = create_segmentation_mask(segmentation);
        cv::Mat scaled_mask = scale_segmentation_mask(mask_image, frame.cols, frame.rows, pad_xy_static);
        cv::addWeighted(result, 1.0 - seg_alpha, scaled_mask, seg_alpha, 0, result);
    }

    return result;
}

// --- Thread function definitions ---

void post_process_thread_func(SafeQueue<std::shared_ptr<DetectionArgs>> *wait_queue,
                              SafeQueue<std::shared_ptr<DisplayArgs>> *display_queue,
                              std::atomic<int> *appQuit) {
    while (appQuit->load() == -1) std::this_thread::sleep_for(std::chrono::microseconds(10));

    while (appQuit->load() == 0 || !wait_queue->empty()) {
        std::shared_ptr<DetectionArgs> args;
        if (!wait_queue->try_pop(args, std::chrono::milliseconds(10))) {
            continue;
        }

        auto yolo_outputs = args->yolo_ie->Wait(args->yolo_request_id);
        auto t_yolo_end = std::chrono::high_resolution_clock::now();

        auto deeplab_outputs = args->deeplab_ie->Wait(args->deeplab_request_id);
        auto t_deeplab_end = std::chrono::high_resolution_clock::now();

        double yolo_inference_time =
            std::chrono::duration<double, std::milli>(t_yolo_end - args->t_yolo_async_start).count();
        double deeplab_inference_time =
            std::chrono::duration<double, std::milli>(t_deeplab_end - args->t_deeplab_async_start).count();

        // Process postprocessing with error handling
        std::vector<YOLOv7Result> detection_results;
        if (!process_yolo_postprocess(*args->ypp, yolo_outputs, detection_results)) {
            appQuit->store(1);
            continue;
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double yolo_postprocess_time = std::chrono::duration<double, std::milli>(t2 - t_yolo_end).count();

        // Process postprocessing with error handling
        DeepLabv3Result segmentation_result;
        if (!process_deeplab_postprocess(*args->dlpp, deeplab_outputs, segmentation_result)) {
            appQuit->store(1);
            continue;
        }
        
        auto t3 = std::chrono::high_resolution_clock::now();
        double deeplab_postprocess_time = std::chrono::duration<double, std::milli>(t3 - t2).count();

        if (args->metrics) {
            update_inflight_metrics(args->metrics, t_deeplab_end);
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
        d_args->t_read = args->t_read;
        d_args->t_yolo_preprocess = args->t_yolo_preprocess;
        d_args->t_deeplab_preprocess = args->t_deeplab_preprocess;
        d_args->t_yolo_inference = yolo_inference_time;
        d_args->t_deeplab_inference = deeplab_inference_time;
        d_args->t_yolo_postprocess = yolo_postprocess_time;
        d_args->t_deeplab_postprocess = deeplab_postprocess_time;
        d_args->metrics = args->metrics;
        d_args->pad_xy = args->pad_xy;
        d_args->ratio = args->ratio;

        while (!display_queue->try_push(d_args)) {
            if (appQuit->load() != 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void handle_display_frame(const cv::Mat& processed_frame, bool is_video_save, bool is_no_show,
                         cv::VideoWriter* writer, std::atomic<int>* appQuit) {
    if (is_video_save) *writer << processed_frame;
    if (!is_no_show) {
        cv::imshow("YOLOv7 + DeepLabV3 Combined Result", processed_frame);
        if (cv::waitKey(1) == 'q') appQuit->store(1);
    }
}

void update_metrics(ProfilingMetrics* metrics, double t_read, double t_yolo_preprocess,
                   double t_deeplab_preprocess, double t_yolo_inference, double t_deeplab_inference,
                   double t_yolo_postprocess, double t_deeplab_postprocess, 
                   double deeplab_render_time, double yolo_render_time, double render_time) {
    std::lock_guard<std::mutex> lock(metrics->metrics_mutex);
    metrics->sum_read += t_read;
    metrics->sum_yolo_preprocess += t_yolo_preprocess;
    metrics->sum_deeplab_preprocess += t_deeplab_preprocess;
    metrics->sum_yolo_inference += t_yolo_inference;
    metrics->sum_deeplab_inference += t_deeplab_inference;
    metrics->sum_yolo_postprocess += t_yolo_postprocess;
    metrics->sum_deeplab_postprocess += t_deeplab_postprocess;
    metrics->sum_deeplab_render += deeplab_render_time;
    metrics->sum_yolo_render += yolo_render_time;
    metrics->sum_render += render_time;
}

void update_inflight_metrics(ProfilingMetrics* metrics, const std::chrono::high_resolution_clock::time_point& t_yolo_end, const std::chrono::high_resolution_clock::time_point& t_deeplab_end) {
    std::lock_guard<std::mutex> lock(metrics->metrics_mutex);
    metrics->infer_last_ts = std::max(t_yolo_end, t_deeplab_end);
    metrics->infer_completed++;
    // Accumulate inflight time before decrementing
    auto now = std::chrono::high_resolution_clock::now();
    metrics->inflight_time_sum +=
        metrics->inflight_current *
        std::chrono::duration<double>(now - metrics->inflight_last_ts).count();
    metrics->inflight_last_ts = now;
    metrics->inflight_current--;
}

void display_thread_func(SafeQueue<std::shared_ptr<DisplayArgs>> *display_queue,
                         std::atomic<int> *appQuit, cv::VideoWriter *writer) {
    while (appQuit->load() == -1) std::this_thread::sleep_for(std::chrono::microseconds(10));

    while (appQuit->load() == 0 || !display_queue->empty()) {
        std::shared_ptr<DisplayArgs> args;
        if (!display_queue->try_pop(args, std::chrono::milliseconds(10))) {
            continue;
        }
        if (!args || !args->original_frame) continue;

        // Draw segmentation first
        auto t_deeplab_render_start = std::chrono::high_resolution_clock::now();
        auto segmentation_result = draw_segmentation(*args->original_frame, args->combined_results->segmentation, args->pad_xy);
        auto t_deeplab_render_end = std::chrono::high_resolution_clock::now();

        // Then draw detections on top
        auto t_yolo_render_start = t_deeplab_render_end;
        auto processed_frame = draw_detections(segmentation_result, args->combined_results->detections, args->pad_xy, args->ratio);
        auto t_yolo_render_end = std::chrono::high_resolution_clock::now();

        if (!processed_frame.empty()) {
            handle_display_frame(processed_frame, args->is_video_save, args->is_no_show, writer, appQuit);
        }

        if (args->processed_count) (*args->processed_count)++;

        auto render_end = std::chrono::high_resolution_clock::now();
        double deeplab_render_time = std::chrono::duration<double, std::milli>(t_deeplab_render_end - t_deeplab_render_start).count();
        double yolo_render_time = std::chrono::duration<double, std::milli>(t_yolo_render_end - t_yolo_render_start).count();
        double render_time = std::chrono::duration<double, std::milli>(render_end - t_yolo_render_end).count();

        if (args->metrics) {
            update_metrics(args->metrics, args->t_read, args->t_yolo_preprocess,
                         args->t_deeplab_preprocess, args->t_yolo_inference, args->t_deeplab_inference,
                         args->t_yolo_postprocess, args->t_deeplab_postprocess, 
                         deeplab_render_time, yolo_render_time, render_time);
        }
    }
}

// --- Performance summary ---

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

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Read" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_read << " ms     " << std::setw(6)
              << std::setprecision(1) << read_fps << " FPS" << std::endl;

    auto print_model_block = [&](const std::string& name, double avg_pre, double pre_fps,
                                 double avg_inf, double inf_fps, double avg_post, double post_fps) {
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

    if (display_on) {
        double avg_yolo_render = metrics.sum_yolo_render / metrics.infer_completed;
        double avg_deeplab_render = metrics.sum_deeplab_render / metrics.infer_completed;
        double avg_render = metrics.sum_render / metrics.infer_completed;
        double yolo_render_fps = avg_yolo_render > 0 ? 1000.0 / avg_yolo_render : 0.0;
        std::cout << " " << std::left << std::setw(15) << "YOLOv7 Render" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(2) << avg_yolo_render << " ms     " << std::setw(6)
                  << std::setprecision(1) << yolo_render_fps << " FPS" << std::endl;
        double deeplab_render_fps = avg_deeplab_render > 0 ? 1000.0 / avg_deeplab_render : 0.0;
        std::cout << " " << std::left << std::setw(15) << "DeepLabV3 Render" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(2) << avg_deeplab_render << " ms     " << std::setw(6)
                  << std::setprecision(1) << deeplab_render_fps << " FPS" << std::endl;
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
              << " :    " << std::fixed << std::setprecision(1) << infer_tp << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Completed"
              << " :    " << metrics.infer_completed << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Inflight Avg"
              << " :    " << std::fixed << std::setprecision(1) << inflight_avg << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Inflight Max"
              << " :      " << metrics.inflight_max << std::endl;
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
    std::string app_name = "YOLOv7 + DeepLabV3 Multi-Model Async Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("y, yolo_model", "YOLOv7 object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(args.yoloModelPath))(
        "d, deeplab_model", "DeepLabV3 segmentation model file (.dxnn, required)",
        cxxopts::value<std::string>(args.deeplabModelPath))(
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
    if (args.yoloModelPath.empty()) {
        std::cerr << "[ERROR] YOLOv7 model path is required. Use -y or --yolo_model option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    if (args.deeplabModelPath.empty()) {
        std::cerr << "[ERROR] DeepLabV3 model path is required. Use -d or --deeplab_model option." << std::endl;
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

// Helper function to submit a frame for async inference
void submit_frame_for_inference(
    const cv::Mat& frame, int& index, int& submitted_frames,
    std::vector<cv::Mat>& images,
    std::vector<std::vector<uint8_t>>& yolo_input_buffers,
    std::vector<std::vector<uint8_t>>& deeplab_input_buffers,
    int yolo_input_height, int yolo_input_width,
    int deeplab_input_height, int deeplab_input_width,
    dxrt::InferenceEngine& yolo_ie, dxrt::InferenceEngine& deeplab_ie,
    YOLOv7PostProcess& yolo_post_processor, DeepLabv3PostProcess& deeplab_post_processor,
    SafeQueue<std::shared_ptr<DetectionArgs>>& wait_queue,
    ProfilingMetrics& profiling_metrics, int& processCount,
    std::atomic<int>& appQuit, bool no_display, bool saveVideo,
    double t_read) {
    
    std::vector<int> yolo_pad_xy{0, 0};
    std::vector<float> yolo_ratio{1.0f, 1.0f};
    std::vector<int> deeplab_pad_xy{0, 0};
    std::vector<float> deeplab_ratio{1.0f, 1.0f};

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::resize(frame, images[index], cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));

    cv::Mat yolo_pre(yolo_input_height, yolo_input_width, CV_8UC3, yolo_input_buffers[index].data());
    make_letterbox_image(images[index], yolo_pre, yolo_pad_xy, yolo_ratio);
    auto t1_yolo = std::chrono::high_resolution_clock::now();

    cv::Mat deeplab_pre(deeplab_input_height, deeplab_input_width, CV_8UC3, deeplab_input_buffers[index].data());
    make_letterbox_image(images[index], deeplab_pre, deeplab_pad_xy, deeplab_ratio);
    auto t1_deeplab = std::chrono::high_resolution_clock::now();

    double yolo_preprocess_time = std::chrono::duration<double, std::milli>(t1_yolo - t0).count();
    double deeplab_preprocess_time = std::chrono::duration<double, std::milli>(t1_deeplab - t1_yolo).count();

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
    args->pad_xy = yolo_pad_xy;
    args->ratio = yolo_ratio;
    args->t_read = t_read;
    args->t_yolo_preprocess = yolo_preprocess_time;
    args->t_deeplab_preprocess = deeplab_preprocess_time;
    args->t_yolo_async_start = t1_yolo;
    args->t_deeplab_async_start = t1_deeplab;
    args->is_no_show = no_display;
    args->is_video_save = saveVideo;

    wait_queue.push(args);

    update_profiling_inflight(profiling_metrics, t1_yolo);

    submitted_frames++;
    if (appQuit.load() == -1) appQuit.store(0);
    index = (index + 1) % ASYNC_BUFFER_SIZE;
}

// Process image frames loop
void process_image_frames(
    const std::vector<std::string>& imageFiles, const std::string& imageFilePath,
    int loopTest, std::vector<cv::Mat>& images,
    std::vector<std::vector<uint8_t>>& yolo_input_buffers,
    std::vector<std::vector<uint8_t>>& deeplab_input_buffers,
    int yolo_input_height, int yolo_input_width,
    int deeplab_input_height, int deeplab_input_width,
    dxrt::InferenceEngine& yolo_ie, dxrt::InferenceEngine& deeplab_ie,
    YOLOv7PostProcess& yolo_post_processor, DeepLabv3PostProcess& deeplab_post_processor,
    SafeQueue<std::shared_ptr<DetectionArgs>>& wait_queue,
    ProfilingMetrics& profiling_metrics, int& processCount, int& index, int& submitted_frames,
    std::atomic<int>& appQuit, bool no_display) {
    
    for (int i = 0; i < loopTest; ++i) {
        if (appQuit.load() > 0) break;
        std::string currentImagePath = imageFiles.empty() ? imageFilePath : imageFiles[i % imageFiles.size()];
        
        auto tr0 = std::chrono::high_resolution_clock::now();
        cv::Mat img = cv::imread(currentImagePath);
        auto tr1 = std::chrono::high_resolution_clock::now();
        double t_read = std::chrono::duration<double, std::milli>(tr1 - tr0).count();

        if (img.empty()) {
            std::cerr << "[ERROR] Failed to read image: " << currentImagePath << std::endl;
            continue;
        }

        submit_frame_for_inference(img, index, submitted_frames, images, yolo_input_buffers, deeplab_input_buffers,
                                   yolo_input_height, yolo_input_width, deeplab_input_height, deeplab_input_width,
                                   yolo_ie, deeplab_ie, yolo_post_processor, deeplab_post_processor,
                                   wait_queue, profiling_metrics, processCount,
                                   appQuit, no_display, false, t_read);
    }
}

// Process video frames loop
void process_video_frames(
    cv::VideoCapture& video,
    std::vector<cv::Mat>& images,
    std::vector<std::vector<uint8_t>>& yolo_input_buffers,
    std::vector<std::vector<uint8_t>>& deeplab_input_buffers,
    int yolo_input_height, int yolo_input_width,
    int deeplab_input_height, int deeplab_input_width,
    dxrt::InferenceEngine& yolo_ie, dxrt::InferenceEngine& deeplab_ie,
    YOLOv7PostProcess& yolo_post_processor, DeepLabv3PostProcess& deeplab_post_processor,
    SafeQueue<std::shared_ptr<DetectionArgs>>& wait_queue,
    ProfilingMetrics& profiling_metrics, int& processCount, int& index, int& submitted_frames,
    std::atomic<int>& appQuit, bool no_display, bool saveVideo) {
    
    bool should_continue = true;
    while (should_continue) {
        cv::Mat frame;
        auto tr0 = std::chrono::high_resolution_clock::now();
        video >> frame;
        auto tr1 = std::chrono::high_resolution_clock::now();
        double t_read = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
        
        if (frame.empty() || appQuit.load() > 0) {
            should_continue = false;
            continue;
        }

        submit_frame_for_inference(frame, index, submitted_frames, images, yolo_input_buffers, deeplab_input_buffers,
                                   yolo_input_height, yolo_input_width, deeplab_input_height, deeplab_input_width,
                                   yolo_ie, deeplab_ie, yolo_post_processor, deeplab_post_processor,
                                   wait_queue, profiling_metrics, processCount,
                                   appQuit, no_display, saveVideo, t_read);
        
        if (appQuit.load() == 1) {
            should_continue = false;
        }
    }
}

// Wait for processing to complete and cleanup threads
void cleanup_threads(const int& processCount, int submitted_frames, std::atomic<int>& appQuit,
                     std::thread& post_thread, std::thread& disp_thread) {
    while (processCount < submitted_frames && appQuit.load() <= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    appQuit.store(1);
    if (post_thread.joinable()) post_thread.join();
    if (disp_thread.joinable()) disp_thread.join();
}

// --- Main function ---

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::atomic<int> appQuit(-1);
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
    dxrt::InferenceEngine yolo_ie(args.yoloModelPath, io);
    dxrt::InferenceEngine deeplab_ie(args.deeplabModelPath, io);

    if (!dxapp::common::minversionforRTandCompiler(&yolo_ie) ||
        !dxapp::common::minversionforRTandCompiler(&deeplab_ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    auto yolo_input_shape = yolo_ie.GetInputs().front().shape();
    auto deeplab_input_shape = deeplab_ie.GetInputs().front().shape();

    auto yolo_input_height = static_cast<int>(yolo_input_shape[1]);
    auto yolo_input_width = static_cast<int>(yolo_input_shape[2]);
    auto deeplab_input_height = static_cast<int>(deeplab_input_shape[1]);
    auto deeplab_input_width = static_cast<int>(deeplab_input_shape[2]);

    auto yolo_post_processor = YOLOv7PostProcess(yolo_input_width, yolo_input_height, 0.25f, 0.25f,
                                                 0.45f, yolo_ie.IsOrtConfigured());
    auto deeplab_post_processor = DeepLabv3PostProcess(deeplab_input_width, deeplab_input_height);

    std::cout << "[INFO] Yolo Model loaded: " << args.yoloModelPath << std::endl;
    std::cout << "[INFO] Yolo Model input size (WxH): " << yolo_input_width << "x" << yolo_input_height << std::endl;
    std::cout << std::endl;

    std::cout << "[INFO] Deeplab Model loaded: " << args.deeplabModelPath << std::endl;
    std::cout << "[INFO] Deeplab Model input size (WxH): " << deeplab_input_width << "x" << deeplab_input_height << std::endl;
    std::cout << std::endl;

    std::vector<std::vector<uint8_t>> yolo_input_buffers(ASYNC_BUFFER_SIZE, std::vector<uint8_t>(yolo_ie.GetInputSize()));
    std::vector<std::vector<uint8_t>> deeplab_input_buffers(ASYNC_BUFFER_SIZE, std::vector<uint8_t>(deeplab_ie.GetInputSize()));

    SafeQueue<std::shared_ptr<DetectionArgs>> wait_queue;
    SafeQueue<std::shared_ptr<DisplayArgs>> display_queue;
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

    std::thread post_thread(post_process_thread_func, &wait_queue, &display_queue, &appQuit);
    std::thread disp_thread(display_thread_func, &display_queue, &appQuit, &writer);

    std::vector<cv::Mat> images(ASYNC_BUFFER_SIZE, cv::Mat(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3));
    int index = 0;
    int submitted_frames = 0;
    auto s_time = std::chrono::high_resolution_clock::now();

    if (is_image) {
        process_image_frames(imageFiles, args.imageFilePath, loopTest, images, yolo_input_buffers, deeplab_input_buffers,
                             yolo_input_height, yolo_input_width, deeplab_input_height, deeplab_input_width,
                             yolo_ie, deeplab_ie, yolo_post_processor, deeplab_post_processor,
                             wait_queue, profiling_metrics, processCount, index, submitted_frames,
                             appQuit, args.no_display);
    } else {
        process_video_frames(video, images, yolo_input_buffers, deeplab_input_buffers,
                             yolo_input_height, yolo_input_width, deeplab_input_height, deeplab_input_width,
                             yolo_ie, deeplab_ie, yolo_post_processor, deeplab_post_processor,
                             wait_queue, profiling_metrics, processCount, index, submitted_frames,
                             appQuit, args.no_display, args.saveVideo);
    }

    cleanup_threads(processCount, submitted_frames, appQuit, post_thread, disp_thread);

    auto e_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(e_time - s_time).count();
    print_performance_summary(profiling_metrics, processCount, total_time, !args.no_display);

    DXRT_TRY_CATCH_END
    return 0;
}
