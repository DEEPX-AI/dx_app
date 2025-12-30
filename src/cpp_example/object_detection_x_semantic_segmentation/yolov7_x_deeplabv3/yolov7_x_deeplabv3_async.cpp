#include <deeplabv3_postprocess.h>
#include <dxrt/dxrt_api.h>
#include <yolov7_postprocess.h>

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

    CombinedResults(const CombinedResults& other)
        : detections(other.detections), segmentation(other.segmentation) {}

    CombinedResults& operator=(const CombinedResults& other) {
        if (this != &other) {
            detections = other.detections;
            segmentation = other.segmentation;
        }
        return *this;
    }

    CombinedResults(CombinedResults&& other) noexcept
        : detections(std::move(other.detections)), segmentation(std::move(other.segmentation)) {}

    CombinedResults& operator=(CombinedResults&& other) noexcept {
        if (this != &other) {
            detections = std::move(other.detections);
            segmentation = std::move(other.segmentation);
        }
        return *this;
    }
};

// Profiling metrics structure for dual models
struct ProfilingMetrics {
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
cv::Mat draw_combined_results(const cv::Mat& frame, const CombinedResults& results,
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
    std::vector<YOLOv7Result> detections_copy = results.detections;
    for (auto& detection : detections_copy) {
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

struct MultiModelArgs {
    CombinedResults* combined_results;
    cv::Mat* current_frame;
    dxrt::InferenceEngine* yolo_ie;
    dxrt::InferenceEngine* deeplab_ie;
    YOLOv7PostProcess* ypp;
    DeepLabv3PostProcess* dlpp;
    std::mutex output_postprocess_lk;
    int* processed_count = nullptr;
    int yolo_request_id = 0;
    int deeplab_request_id = 0;
    bool is_no_show = false;
    bool is_video_save = false;

    // Timing information
    double t_yolo_preprocess = 0.0;
    double t_deeplab_preprocess = 0.0;
    std::chrono::high_resolution_clock::time_point t_yolo_async_start;
    std::chrono::high_resolution_clock::time_point t_deeplab_async_start;
    ProfilingMetrics* metrics = nullptr;

    MultiModelArgs()
        : combined_results(nullptr),
          current_frame(nullptr),
          yolo_ie(nullptr),
          deeplab_ie(nullptr),
          ypp(nullptr),
          dlpp(nullptr),
          processed_count(nullptr),
          yolo_request_id(0),
          deeplab_request_id(0),
          is_no_show(false),
          is_video_save(false),
          t_yolo_preprocess(0.0),
          t_deeplab_preprocess(0.0),
          metrics(nullptr) {}

    ~MultiModelArgs() {
        if (combined_results != nullptr) {
            delete combined_results;
            combined_results = nullptr;
        }
    }
};

struct MultiModelDisplayArgs {
    CombinedResults* combined_results;
    cv::Mat* original_frame;
    YOLOv7PostProcess* ypp;
    DeepLabv3PostProcess* dlpp;
    std::mutex display_lk;
    bool is_no_show = false;
    bool is_video_save = false;
    int* processed_count = nullptr;

    // Timing information
    double t_yolo_preprocess = 0.0;
    double t_deeplab_preprocess = 0.0;
    double t_yolo_inference = 0.0;
    double t_deeplab_inference = 0.0;
    double t_yolo_postprocess = 0.0;
    double t_deeplab_postprocess = 0.0;
    ProfilingMetrics* metrics = nullptr;

    MultiModelDisplayArgs()
        : combined_results(nullptr),
          original_frame(nullptr),
          ypp(nullptr),
          dlpp(nullptr),
          is_no_show(false),
          is_video_save(false),
          processed_count(nullptr),
          t_yolo_preprocess(0.0),
          t_deeplab_preprocess(0.0),
          t_yolo_inference(0.0),
          t_deeplab_inference(0.0),
          t_yolo_postprocess(0.0),
          t_deeplab_postprocess(0.0),
          metrics(nullptr) {}

    ~MultiModelDisplayArgs() {
        if (combined_results != nullptr) {
            delete combined_results;
            combined_results = nullptr;
        }
    }
};

// Thread-safe queue wrapper for multi-model processing
class MultiModelSafeQueue {
   private:
    std::queue<MultiModelArgs*> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;

   public:
    MultiModelSafeQueue(size_t max_size = MAX_QUEUE_SIZE) : max_size_(max_size) {}

    void push(MultiModelArgs* item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return queue_.size() < max_size_; });
        queue_.push(item);
        condition_.notify_one();
    }

    MultiModelArgs* pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        MultiModelArgs* item = queue_.front();
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

void multi_model_post_process_thread_func(MultiModelSafeQueue* wait_queue,
                                          std::queue<MultiModelDisplayArgs*>* display_queue,
                                          std::atomic<int>* appQuit) {
    while (appQuit->load() == -1) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    std::cout << "[DXAPP] [INFO] multi-model post processing thread start" << std::endl;

    std::vector<std::unique_ptr<MultiModelDisplayArgs>> display_args_list;
    display_args_list.reserve(ASYNC_BUFFER_SIZE);

    while (appQuit->load() == 0) {
        if (wait_queue->size() > 0) {
            MultiModelArgs* args = wait_queue->pop();
            auto display_args = std::unique_ptr<MultiModelDisplayArgs>(new MultiModelDisplayArgs());

            // Wait for both model outputs
            auto yolo_outputs = args->yolo_ie->Wait(args->yolo_request_id);
            auto t_yolo_end = std::chrono::high_resolution_clock::now();
            
            // Calculate inference times
            auto deeplab_outputs = args->deeplab_ie->Wait(args->deeplab_request_id);
            auto t_deeplab_end = std::chrono::high_resolution_clock::now();

            double yolo_inference_time =
                std::chrono::duration<double, std::milli>(t_yolo_end - args->t_yolo_async_start)
                    .count();
            double deeplab_inference_time = std::chrono::duration<double, std::milli>(
                                                t_deeplab_end - args->t_deeplab_async_start)
                                                .count();

            // Start postprocess timing for YOLO
            auto t1 = std::chrono::high_resolution_clock::now();
            auto detection_results = args->ypp->postprocess(yolo_outputs);
            auto t2 = std::chrono::high_resolution_clock::now();
            double yolo_postprocess_time =
                std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Start postprocess timing for DeepLab
            auto segmentation_result = args->dlpp->postprocess(deeplab_outputs);
            auto t3 = std::chrono::high_resolution_clock::now();
            double deeplab_postprocess_time =
                std::chrono::duration<double, std::milli>(t3 - t2).count();

            // Update inflight tracking
            if (args->metrics) {
                std::unique_lock<std::mutex> metrics_lock(args->metrics->metrics_mutex);

                args->metrics->infer_last_ts = std::max(t_yolo_end, t_deeplab_end);
                args->metrics->infer_completed++;

                // Update inflight tracking - decrease current count
                auto dt = std::chrono::duration<double>(args->metrics->infer_last_ts -
                                                        args->metrics->inflight_last_ts)
                              .count();
                args->metrics->inflight_time_sum += args->metrics->inflight_current * dt;
                args->metrics->inflight_last_ts = args->metrics->infer_last_ts;
                args->metrics->inflight_current--;
            }

            // Combine results
            CombinedResults combined(detection_results, segmentation_result);

            {
                std::unique_lock<std::mutex> lock(args->output_postprocess_lk);
                display_args->combined_results = new CombinedResults(combined);
                display_args->original_frame = new cv::Mat(args->current_frame->clone());
                display_args->processed_count = args->processed_count;
                display_args->ypp = args->ypp;
                display_args->dlpp = args->dlpp;
                display_args->is_no_show = args->is_no_show;
                display_args->is_video_save = args->is_video_save;

                // Copy timing information
                display_args->t_yolo_preprocess = args->t_yolo_preprocess;
                display_args->t_deeplab_preprocess = args->t_deeplab_preprocess;
                display_args->t_yolo_inference = yolo_inference_time;
                display_args->t_deeplab_inference = deeplab_inference_time;
                display_args->t_yolo_postprocess = yolo_postprocess_time;
                display_args->t_deeplab_postprocess = deeplab_postprocess_time;
                display_args->metrics = args->metrics;

                display_queue->push(display_args.get());
                display_args_list.push_back(std::move(display_args));
            }
        }
    }
}

void multi_model_display_thread_func(std::queue<MultiModelDisplayArgs*>* display_queue,
                                     std::atomic<int>* appQuit, cv::VideoWriter* writer,
                                     std::vector<int>* pad_xy, float* scale_factor) {
    while (appQuit->load() == -1) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    std::cout << "[DXAPP] [INFO] multi-model display thread start" << std::endl;
    while (appQuit->load() == 0) {
        if (!display_queue->empty()) {
            auto args = display_queue->front();
            display_queue->pop();

            // Start render timing
            auto render_start = std::chrono::high_resolution_clock::now();

            // Draw combined results (segmentation + detection)
            auto processed_frame = draw_combined_results(
                *args->original_frame, *args->combined_results, *pad_xy, *scale_factor);

            // Calculate render time
            auto render_end = std::chrono::high_resolution_clock::now();
            double render_time =
                std::chrono::duration<double, std::milli>(render_end - render_start).count();

            {
                std::unique_lock<std::mutex> lock(args->display_lk);

                // Update metrics with timing information
                if (args->metrics) {
                    std::unique_lock<std::mutex> metrics_lock(args->metrics->metrics_mutex);
                    args->metrics->sum_yolo_preprocess += args->t_yolo_preprocess;
                    args->metrics->sum_deeplab_preprocess += args->t_deeplab_preprocess;
                    args->metrics->sum_yolo_inference += args->t_yolo_inference;
                    args->metrics->sum_deeplab_inference += args->t_deeplab_inference;
                    args->metrics->sum_yolo_postprocess += args->t_yolo_postprocess;
                    args->metrics->sum_deeplab_postprocess += args->t_deeplab_postprocess;
                    args->metrics->sum_render += render_time;
                }

                if (args->combined_results != nullptr) {
                    delete args->combined_results;
                    args->combined_results = nullptr;
                }
                (*args->processed_count)++;

                if (processed_frame.dims == 0) {
                    continue;
                }
                if (args->is_video_save) {
                    *writer << processed_frame;
                }
                if (!args->is_no_show) {
                    cv::imshow("YOLOv7 + DeepLabV3 Combined Result", processed_frame);
                    if (cv::waitKey(1) == 'q') {
                        args->is_no_show = true;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::atomic<int> appQuit(-1);
    std::string yoloModelPath = "", deeplabModelPath = "", imgFile = "", videoFile = "";
    bool cameraMode = false, fps_only = false, saveVideo = false;
    int loopTest = 1, loopCount = 1, processCount = 0;

    std::string app_name = "YOLOv7 + DeepLabV3 Multi-Model Async Example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("y, yolo_model", "YOLOv7 object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(yoloModelPath))(
        "d, deeplab_model", "DeepLabV3 segmentation model file (.dxnn, required)",
        cxxopts::value<std::string>(deeplabModelPath))(
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
    if (!cameraMode && imgFile.empty() && videoFile.empty()) {
        std::cerr << "[ERROR] Image path or video path is required. Use -i or "
                     "--image_path option or -v or --video_path option."
                  << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    loopCount = loopTest;

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

    // Get model input dimensions (assuming both models have same input size)
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
    std::vector<std::vector<uint8_t>> yolo_input_buffers(ASYNC_BUFFER_SIZE);
    std::vector<std::vector<uint8_t>> yolo_output_buffers(ASYNC_BUFFER_SIZE);
    std::vector<std::vector<uint8_t>> deeplab_input_buffers(ASYNC_BUFFER_SIZE);
    std::vector<std::vector<uint8_t>> deeplab_output_buffers(ASYNC_BUFFER_SIZE);

    for (auto& input_buffer : yolo_input_buffers) {
        input_buffer = std::vector<uint8_t>(yolo_ie.GetInputSize());
    }
    for (auto& output_buffer : yolo_output_buffers) {
        output_buffer = std::vector<uint8_t>(yolo_ie.GetOutputSize());
    }
    for (auto& input_buffer : deeplab_input_buffers) {
        input_buffer = std::vector<uint8_t>(deeplab_ie.GetInputSize());
    }
    for (auto& output_buffer : deeplab_output_buffers) {
        output_buffer = std::vector<uint8_t>(deeplab_ie.GetOutputSize());
    }

    MultiModelSafeQueue wait_queue;
    std::queue<MultiModelDisplayArgs*> display_queue;
    ProfilingMetrics profiling_metrics;

    std::thread post_process_thread(multi_model_post_process_thread_func, &wait_queue,
                                    &display_queue, &appQuit);
    cv::VideoWriter writer;
    std::vector<int> yolo_pad_xy{0, 0};
    float yolo_scale_factor = 1.f;
    std::vector<int> deeplabv3_pad_xy{0, 0};

    std::thread display_thread(multi_model_display_thread_func, &display_queue, &appQuit, &writer,
                               &yolo_pad_xy, &yolo_scale_factor);

    // Image processing
    if (!imgFile.empty()) {
        if (loopTest == 1) {
            std::cout << "loopTest is set to 100 when an image file is provided." << std::endl;
            loopCount = 100;
            loopTest = 100;
        }

        cv::Mat image = cv::imread(imgFile, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Image file is not valid." << std::endl;
            exit(1);
        }

        std::cout << "[INFO] Image resolution (WxH): " << image.cols << "x" << image.rows
                  << std::endl;
        std::cout << "[INFO] Total frames: " << loopCount << std::endl;

        cv::resize(image, image, cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H),
                   cv::INTER_LINEAR);

        yolo_scale_factor = std::min(yolo_input_width / static_cast<float>(image.cols),
                                     yolo_input_height / static_cast<float>(image.rows));
        int letterbox_pad_x =
            std::max(0.f, (yolo_input_width - image.cols * yolo_scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (yolo_input_height - image.rows * yolo_scale_factor) / 2);
        yolo_pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();

        std::vector<std::unique_ptr<MultiModelArgs>> multi_model_args_list;
        multi_model_args_list.reserve(ASYNC_BUFFER_SIZE);
        int index = 0;

        do {
            index = (index + 1) % ASYNC_BUFFER_SIZE;

            // Preprocess for YOLOv7
            auto t0_yolo = std::chrono::high_resolution_clock::now();
            cv::Mat yolo_preprocessed_image = cv::Mat(yolo_input_height, yolo_input_width, CV_8UC3,
                                                      yolo_input_buffers[index].data());
            make_letterbox_image(image, yolo_preprocessed_image, cv::COLOR_BGR2RGB, yolo_pad_xy);
            auto t1_yolo = std::chrono::high_resolution_clock::now();
            double yolo_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_yolo - t0_yolo).count();

            // Preprocess for DeepLabV3
            auto t0_deeplab = std::chrono::high_resolution_clock::now();
            cv::Mat deeplab_preprocessed_image =
                cv::Mat(deeplab_input_height, deeplab_input_width, CV_8UC3,
                        deeplab_input_buffers[index].data());
            std::vector<int> deeplab_pad_xy = {0, 0};  // Assuming same padding for simplicity
            make_letterbox_image(image, deeplab_preprocessed_image, cv::COLOR_BGR2RGB,
                                 deeplab_pad_xy);
            auto t1_deeplab = std::chrono::high_resolution_clock::now();
            double deeplab_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_deeplab - t0_deeplab).count();

            // Start async inference for both models
            auto yolo_req_id = yolo_ie.RunAsync(yolo_preprocessed_image.data, nullptr,
                                                yolo_output_buffers[index].data());
            auto deeplab_req_id = deeplab_ie.RunAsync(deeplab_preprocessed_image.data, nullptr,
                                                      deeplab_output_buffers[index].data());
            auto t2 = std::chrono::high_resolution_clock::now();

            // Update inflight tracking
            {
                std::unique_lock<std::mutex> metrics_lock(profiling_metrics.metrics_mutex);
                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1_yolo;
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

            // Create MultiModelArgs
            auto args = std::unique_ptr<MultiModelArgs>(new MultiModelArgs());
            args->yolo_ie = &yolo_ie;
            args->deeplab_ie = &deeplab_ie;
            args->ypp = &yolo_post_processor;
            args->dlpp = &deeplab_post_processor;
            args->current_frame = &image;
            args->combined_results = nullptr;
            args->yolo_request_id = yolo_req_id;
            args->deeplab_request_id = deeplab_req_id;
            args->processed_count = &processCount;
            args->is_no_show = fps_only;
            args->is_video_save = saveVideo;
            args->t_yolo_preprocess = yolo_preprocess_time;
            args->t_deeplab_preprocess = deeplab_preprocess_time;
            args->t_yolo_async_start = t1_yolo;
            args->t_deeplab_async_start = t1_deeplab;
            args->metrics = &profiling_metrics;

            wait_queue.push(args.get());
            multi_model_args_list.push_back(std::move(args));
            if (appQuit.load() == -1) {
                appQuit.store(0);
            }
        } while (--loopTest);

        // Wait for all processing to complete
        while (processCount < loopCount) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        appQuit.store(1);
        post_process_thread.join();
        display_thread.join();

        auto e = std::chrono::high_resolution_clock::now();

        // Print profiling information
        if (profiling_metrics.infer_completed > 0) {
            double avg_yolo_pre =
                profiling_metrics.sum_yolo_preprocess / profiling_metrics.infer_completed;
            double avg_deeplab_pre =
                profiling_metrics.sum_deeplab_preprocess / profiling_metrics.infer_completed;
            double avg_yolo_inf =
                profiling_metrics.sum_yolo_inference / profiling_metrics.infer_completed;
            double avg_deeplab_inf =
                profiling_metrics.sum_deeplab_inference / profiling_metrics.infer_completed;
            double avg_yolo_post =
                profiling_metrics.sum_yolo_postprocess / profiling_metrics.infer_completed;
            double avg_deeplab_post =
                profiling_metrics.sum_deeplab_postprocess / profiling_metrics.infer_completed;

            auto inflight_time_window =
                std::chrono::duration<double>(profiling_metrics.infer_last_ts -
                                              profiling_metrics.infer_first_ts)
                    .count();
            double infer_tp = profiling_metrics.infer_completed / inflight_time_window;
            double inflight_avg = profiling_metrics.inflight_time_sum / inflight_time_window;

            std::cout
                << "[PROFILE] ======== YOLOv7 + DeepLabV3 Multi-Model Performance Report ========"
                << std::endl;
            std::cout << "[PROFILE] YOLOv7 Preprocess latency = " << std::fixed
                      << std::setprecision(2) << avg_yolo_pre << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_yolo_pre) << " FPS)" << std::endl;
            std::cout << "[PROFILE] DeepLabV3 Preprocess latency = " << std::setprecision(2)
                      << avg_deeplab_pre << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_deeplab_pre) << " FPS)" << std::endl;
            std::cout << "[PROFILE] YOLOv7 Inference latency = " << std::setprecision(2)
                      << avg_yolo_inf << " ms (" << std::setprecision(1) << (1000.0 / avg_yolo_inf)
                      << " FPS)" << std::endl;
            std::cout << "[PROFILE] DeepLabV3 Inference latency = " << std::setprecision(2)
                      << avg_deeplab_inf << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_deeplab_inf) << " FPS)" << std::endl;
            std::cout << "[PROFILE] Combined throughput = " << std::setprecision(1) << infer_tp
                      << " FPS | Inflight avg = " << std::setprecision(1) << inflight_avg
                      << ", max = " << profiling_metrics.inflight_max << std::endl;
            std::cout << "[PROFILE] YOLOv7 Postprocess latency = " << std::setprecision(2)
                      << avg_yolo_post << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_yolo_post) << " FPS)" << std::endl;
            std::cout << "[PROFILE] DeepLabV3 Postprocess latency = " << std::setprecision(2)
                      << avg_deeplab_post << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_deeplab_post) << " FPS)" << std::endl;

            if (!fps_only) {
                double avg_render =
                    profiling_metrics.sum_render / profiling_metrics.infer_completed;
                double render_fps = 1000.0 / avg_render;
                std::cout << "[PROFILE] Combined Render latency = " << std::setprecision(2)
                          << avg_render << " ms (" << std::setprecision(1) << render_fps << " FPS)"
                          << std::endl;
            }
        }

        double overall_fps =
            processCount /
            (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0);
        std::cout << "[SUCCESS] End-to-End Combined FPS : " << std::setprecision(1) << overall_fps
                  << std::endl;
    } else if (!videoFile.empty()) {
        std::cout << "loopTest is set to 1 when a video file is provided." << std::endl;
        loopTest = 1;
        cv::VideoCapture video(videoFile);
        if (!video.isOpened()) {
            std::cerr << "[ERROR] Video file is not valid." << std::endl;
            exit(1);
        }

        // Print video information
        int video_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
        int video_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
        double video_fps = video.get(cv::CAP_PROP_FPS);

        std::cout << "[INFO] Video resolution (WxH): " << video_width << "x" << video_height
                  << std::endl;
        std::cout << "[INFO] Total frames: " << total_frames << std::endl;
        std::cout << "[INFO] Video FPS: " << std::fixed << std::setprecision(2) << video_fps
                  << std::endl;

        if (fps_only) {
            std::cout << "Processing video stream... Only FPS will be displayed." << std::endl;
        }
        if (saveVideo) {
            writer = cv::VideoWriter("result.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                     video.get(cv::CAP_PROP_FPS),
                                     cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            if (!writer.isOpened()) {
                std::cerr << "[ERROR] Failed to open video writer." << std::endl;
                exit(1);
            }
        }
        loopCount = video.get(cv::CAP_PROP_FRAME_COUNT);
        
        yolo_scale_factor = std::min(yolo_input_width / static_cast<float>(SHOW_WINDOW_SIZE_W),
                                     yolo_input_height / static_cast<float>(SHOW_WINDOW_SIZE_H));
        int letterbox_pad_x =
            std::max(0.f, (yolo_input_width - SHOW_WINDOW_SIZE_W * yolo_scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (yolo_input_height - SHOW_WINDOW_SIZE_H * yolo_scale_factor) / 2);
        yolo_pad_xy = {letterbox_pad_x, letterbox_pad_y};

        auto s = std::chrono::high_resolution_clock::now();

        std::vector<std::unique_ptr<MultiModelArgs>> multi_model_args_list;
        multi_model_args_list.reserve(ASYNC_BUFFER_SIZE);
        std::vector<cv::Mat> images(ASYNC_BUFFER_SIZE);
        for (auto& img : images) {
            img = cv::Mat(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3);
        }
        int index = 0;
        do {
            index = (index + 1) % ASYNC_BUFFER_SIZE;
            cv::Mat frame;
            video >> frame;
            if (frame.empty()) {
                break;
            }

            // Start preprocess timing
            cv::resize(frame, images[index], cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H),
                       cv::INTER_LINEAR);
            auto t0_yolo = std::chrono::high_resolution_clock::now();
            cv::Mat yolo_preprocessed_image = cv::Mat(yolo_post_processor.get_input_height(),
                                                      yolo_post_processor.get_input_width(),
                                                      CV_8UC3, yolo_input_buffers[index].data());
            make_letterbox_image(images[index], yolo_preprocessed_image, cv::COLOR_BGR2RGB,
                                 yolo_pad_xy);
            auto t1_yolo = std::chrono::high_resolution_clock::now();
            double yolo_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_yolo - t0_yolo).count();

            auto t0_deeplab = std::chrono::high_resolution_clock::now();
            cv::Mat deeplabv3_preprocessed_image = cv::Mat(
                deeplab_post_processor.get_input_height(), deeplab_post_processor.get_input_width(),
                CV_8UC3, deeplab_input_buffers[index].data());
            make_letterbox_image(images[index], deeplabv3_preprocessed_image, cv::COLOR_BGR2RGB,
                                 deeplabv3_pad_xy);
            auto t1_deeplab = std::chrono::high_resolution_clock::now();
            double deeplab_preprocess_time =
                std::chrono::duration<double, std::milli>(t1_deeplab - t0_deeplab).count();

            auto yolo_req_id = yolo_ie.RunAsync(yolo_preprocessed_image.data, nullptr,
                                                yolo_output_buffers[index].data());
            auto deeplab_req_id = deeplab_ie.RunAsync(deeplabv3_preprocessed_image.data, nullptr,
                                                   deeplab_output_buffers[index].data());
            auto t2 = std::chrono::high_resolution_clock::now();

            // Update inflight tracking
            {
                std::unique_lock<std::mutex> metrics_lock(profiling_metrics.metrics_mutex);

                if (profiling_metrics.first_inference) {
                    profiling_metrics.infer_first_ts = t1_yolo;
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

            // Create MultiModelArgs
            auto args = std::unique_ptr<MultiModelArgs>(new MultiModelArgs());
            args->yolo_ie = &yolo_ie;
            args->deeplab_ie = &deeplab_ie;
            args->ypp = &yolo_post_processor;
            args->dlpp = &deeplab_post_processor;
            args->current_frame = &images[index];
            args->combined_results = nullptr;
            args->yolo_request_id = yolo_req_id;
            args->deeplab_request_id = deeplab_req_id;
            args->processed_count = &processCount;
            args->is_no_show = fps_only;
            args->is_video_save = saveVideo;
            args->t_yolo_preprocess = yolo_preprocess_time;
            args->t_deeplab_preprocess = deeplab_preprocess_time;
            args->t_yolo_async_start = t1_yolo;
            args->t_deeplab_async_start = t1_deeplab;
            args->metrics = &profiling_metrics;

            wait_queue.push(args.get());
            multi_model_args_list.push_back(std::move(args));
            if (appQuit.load() == -1) {
                appQuit.store(0);
            }
        } while (true);

        // Wait for all processing to complete
        while (processCount < loopCount) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        appQuit.store(1);
        post_process_thread.join();
        display_thread.join();

        auto e = std::chrono::high_resolution_clock::now();

        // Print profiling information similar to Python version
        if (profiling_metrics.infer_completed > 0) {
            double avg_yolo_pre =
                profiling_metrics.sum_yolo_preprocess / profiling_metrics.infer_completed;
            double avg_deeplab_pre =
                profiling_metrics.sum_deeplab_preprocess / profiling_metrics.infer_completed;
            double avg_yolo_inf =
                profiling_metrics.sum_yolo_inference / profiling_metrics.infer_completed;
            double avg_deeplab_inf =
                profiling_metrics.sum_deeplab_inference / profiling_metrics.infer_completed;
            double avg_yolo_post =
                profiling_metrics.sum_yolo_postprocess / profiling_metrics.infer_completed;
            double avg_deeplab_post =
                profiling_metrics.sum_deeplab_postprocess / profiling_metrics.infer_completed;

            auto inflight_time_window =
                std::chrono::duration<double>(profiling_metrics.infer_last_ts -
                                              profiling_metrics.infer_first_ts)
                    .count();
            double infer_tp = profiling_metrics.infer_completed / inflight_time_window;
            double inflight_avg = profiling_metrics.inflight_time_sum / inflight_time_window;

            std::cout
                << "[PROFILE] ======== YOLOv7 + DeepLabV3 Multi-Model Performance Report ========"
                << std::endl;
            std::cout << "[PROFILE] YOLOv7 Preprocess latency = " << std::fixed
                      << std::setprecision(2) << avg_yolo_pre << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_yolo_pre) << " FPS)" << std::endl;
            std::cout << "[PROFILE] DeepLabV3 Preprocess latency = " << std::setprecision(2)
                      << avg_deeplab_pre << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_deeplab_pre) << " FPS)" << std::endl;
            std::cout << "[PROFILE] YOLOv7 Inference latency = " << std::setprecision(2)
                      << avg_yolo_inf << " ms (" << std::setprecision(1) << (1000.0 / avg_yolo_inf)
                      << " FPS)" << std::endl;
            std::cout << "[PROFILE] DeepLabV3 Inference latency = " << std::setprecision(2)
                      << avg_deeplab_inf << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_deeplab_inf) << " FPS)" << std::endl;
            std::cout << "[PROFILE] Combined throughput = " << std::setprecision(1) << infer_tp
                      << " FPS | Inflight avg = " << std::setprecision(1) << inflight_avg
                      << ", max = " << profiling_metrics.inflight_max << std::endl;
            std::cout << "[PROFILE] YOLOv7 Postprocess latency = " << std::setprecision(2)
                      << avg_yolo_post << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_yolo_post) << " FPS)" << std::endl;
            std::cout << "[PROFILE] DeepLabV3 Postprocess latency = " << std::setprecision(2)
                      << avg_deeplab_post << " ms (" << std::setprecision(1)
                      << (1000.0 / avg_deeplab_post) << " FPS)" << std::endl;

            if (!fps_only) {
                double avg_render =
                    profiling_metrics.sum_render / profiling_metrics.infer_completed;
                double render_fps = 1000.0 / avg_render;
                std::cout << "[PROFILE] Combined Render latency = " << std::setprecision(2)
                          << avg_render << " ms (" << std::setprecision(1) << render_fps << " FPS)"
                          << std::endl;
            }
        }

        double overall_fps =
            processCount /
            (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0);
        std::cout << "[SUCCESS] End-to-End Combined FPS : " << std::setprecision(1) << overall_fps
                  << std::endl;
    }

    std::cout << "\nMulti-model example completed successfully!" << std::endl;
    DXRT_TRY_CATCH_END
    return 0;
}