/**
 * @file async_detection_runner.hpp
 * @brief Asynchronous inference runner using factory pattern
 * 
 * Provides a generic async runner that accepts any factory implementation.
 */

#ifndef ASYNC_RUNNER_HPP
#define ASYNC_RUNNER_HPP

#include <dxrt/dxrt_api.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cxxopts.hpp>
#include <experimental/filesystem>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <set>
#include <thread>
#include <vector>

#include "common/base/i_factory.hpp"
#include "common/config/model_config.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/verify_serialize.hpp"

namespace dxapp {

constexpr size_t SHOW_WINDOW_SIZE_W = 960;
constexpr size_t SHOW_WINDOW_SIZE_H = 640;
constexpr size_t ASYNC_BUFFER_SIZE = 40;
constexpr size_t ASYNC_MAX_QUEUE_SIZE = 100;

// Command line arguments structure (duplicated from sync_runner for independent compilation)
struct CommandLineArgs {
    std::string modelPath;
    std::string imageFilePath;
    std::string videoFile;
    std::string rtspUrl;
    std::string saveDir;
    std::string configPath;
    int cameraIndex = -1;
    int loopTest = -1;
    bool no_display = false;
    bool saveMode = false;
    bool dumpTensors = false;
    bool verbose = false;
};

// Async profiling metrics structure
struct AsyncProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    double sum_postprocess = 0.0;
    double sum_render = 0.0;
    double sum_save = 0.0;
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
    std::condition_variable inflight_cv;  // back-pressure signaling

    /** Block until inflight count drops below ASYNC_BUFFER_SIZE. */
    void waitForSlot() {
        std::unique_lock<std::mutex> lock(metrics_mutex);
        inflight_cv.wait(lock, [this] { return inflight_current < static_cast<int>(ASYNC_BUFFER_SIZE); });
    }

    /** Notify producer that a slot is available. Call after decrementing inflight_current. */
    void notifySlot() {
        inflight_cv.notify_one();
    }
};

template <typename T>
class SafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;
    std::atomic<bool> stopped_{false};

public:
    explicit SafeQueue(size_t max_size = ASYNC_MAX_QUEUE_SIZE) : max_size_(max_size) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.size() >= max_size_ && !stopped_.load(std::memory_order_relaxed)) {
            condition_.wait_for(lock, std::chrono::milliseconds(100));
        }
        if (stopped_.load(std::memory_order_relaxed)) return false;
        queue_.push(std::move(item));
        condition_.notify_one();
        return true;
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        condition_.notify_one();
        return item;
    }

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

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void shutdown() {
        stopped_.store(true, std::memory_order_relaxed);
        condition_.notify_all();
    }
};

// User data passed through RunAsync callback
struct AsyncUserData {
    cv::Mat display_frame;
    PreprocessContext ctx;
    std::string save_path;  // Empty means no save for this frame
    std::chrono::high_resolution_clock::time_point submit_ts;  // For inference time measurement

    AsyncUserData() = default;
    AsyncUserData(const AsyncUserData&) = default;
    AsyncUserData& operator=(const AsyncUserData&) = default;
    AsyncUserData(AsyncUserData&&) noexcept = default;
    AsyncUserData& operator=(AsyncUserData&&) noexcept = default;
    ~AsyncUserData() = default;
};

// Display arguments for async processing
struct AsyncDisplayArgs {
    std::shared_ptr<std::vector<DetectionResult>> detections;
    std::shared_ptr<cv::Mat> original_frame;
    std::string save_path;
    double t_read = 0.0;
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    PreprocessContext ctx;
    AsyncDisplayArgs() = default;
    AsyncDisplayArgs(const AsyncDisplayArgs&) = default;
    AsyncDisplayArgs& operator=(const AsyncDisplayArgs&) = default;
    AsyncDisplayArgs(AsyncDisplayArgs&&) noexcept = default;
    AsyncDisplayArgs& operator=(AsyncDisplayArgs&&) noexcept = default;
};

/**
 * @brief Generic asynchronous runner for detection-based models
 * @tparam FactoryT The factory type (must derive from IDetectionFactory)
 */
template <typename FactoryT>
class AsyncDetectionRunner {
    bool verbose_ = false;

public:
    explicit AsyncDetectionRunner(std::unique_ptr<FactoryT> factory)
        : factory_(std::move(factory)) {}

    int run(int argc, char* argv[]) {
        DXRT_TRY_CATCH_BEGIN
        installSignalHandlers();
        int processCount = 0;

        // Parse command line arguments
        CommandLineArgs args = parseCommandLine(argc, argv);
        verbose_ = args.verbose;
        validateArguments(args);

        // Reconstruct command line for run_info.txt
        std::string command_line;
        for (int i = 0; i < argc; ++i) {
            if (i > 0) command_line += " ";
            command_line += argv[i];
        }

        // Handle image file or directory
        std::vector<std::string> imageFiles;
        bool is_image = !args.imageFilePath.empty();
        int loopTest = args.loopTest;
        if (is_image) {
            auto result = processImagePath(args.imageFilePath, loopTest);
            imageFiles = result.first;
            loopTest = result.second;
        } else if (loopTest == -1) {
            loopTest = 1;
        }

        // Initialize inference engine (Legacy API)
        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(args.modelPath, io);
        model_path_ = args.modelPath;

        // Version compatibility check (matching Legacy)
        if (!dxapp::minversionforRTandCompiler(&ie)) {
            std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                         "compatible with the version of the runtime. Please compile the model again."
                      << std::endl;
            return -1;
        }

        // Get input dimensions (Legacy API)
        auto input_shape = ie.GetInputs().front().shape();
        int input_height, input_width;
        parseInputShape(input_shape, input_width, input_height);

        // Detect float input model and layout
        bool is_float_input = (ie.GetInputs().front().type() == dxrt::DataType::FLOAT);
        bool is_nhwc = isInputNHWC(input_shape);

        // Create processors and visualizer using factory
        // Load model configuration if provided
        if (!args.configPath.empty()) {
            dxapp::ModelConfig config(args.configPath);
            factory_->loadConfig(config);
        }

        auto preprocessor = factory_->createPreprocessor(input_width, input_height);
        // Use shared_ptr so the callback lambda can capture by value and extend lifetime
        // past ie.Wait() which returns before the background callback thread fires.
        std::shared_ptr<IPostprocessor<DetectionResult>> postprocessor(
            factory_->createPostprocessor(input_width, input_height, ie.IsOrtConfigured()));
        auto visualizer = factory_->createVisualizer();

        std::cout << "[INFO] Model loaded: " << args.modelPath << std::endl;
        std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height << std::endl;
        std::cout << std::endl;

        // Allocate pre-allocated input buffers (matching Legacy: ASYNC_BUFFER_SIZE buffers)
        size_t input_size = ie.GetInputSize();
        std::vector<std::vector<uint8_t>> input_buffers(ASYNC_BUFFER_SIZE, std::vector<uint8_t>(input_size));

        cv::VideoCapture video;
        cv::VideoWriter writer;
        std::string run_dir;            // Populated when saveMode or dumpTensors
        std::string image_save_dir;     // For image mode save
        std::string video_save_path;    // For video mode save
        std::set<std::string, std::less<>> saved_files;  // Dedup for image save in loop mode

        // Determine input source string for run_info.txt
        std::string input_source_str = buildInputSourceString(
            args.imageFilePath, args.videoFile, args.cameraIndex, args.rtspUrl);

        if (is_image) {
            // Image Save Setup
            if (args.saveMode) {
                std::string run_kind = fs::is_directory(args.imageFilePath) ? "image-dir" : "image";
                std::string run_name = fs::path(args.imageFilePath).filename().string();
                run_dir = makeRunDir(args.saveDir, factory_->getModelName() + "_async",
                                     run_kind, run_name);
                fs::create_directories(run_dir);
                image_save_dir = run_dir;
                writeRunInfo(run_dir, argv[0], args.modelPath, input_source_str);
            }
        } else {
            // Open video capture
            if (!openVideoCapture(video, args)) {
                std::cerr << "[ERROR] Failed to open input source." << std::endl;
                return -1;
            }

            // Print video source info (matching Legacy)
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
            }

            if (args.verbose) {
                std::cout << "[INFO] Loop count: " << loopTest << std::endl;
                std::cout << "[INFO] " << source_info << std::endl;
                std::cout << "[INFO] Input source resolution (WxH): " << frame_width << "x" << frame_height << std::endl;
                std::cout << "[INFO] Input source FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
            }
            if (!args.videoFile.empty()) {
                if (args.verbose) {
                    std::cout << "[INFO] Total frames: " << total_frames << std::endl;
                }
            }
            std::cout << std::endl;

            // Video Save Setup with run_dir + mp4v→XVID fallback
            if (args.saveMode) {
                std::string run_name;
                if (args.cameraIndex >= 0) {
                    run_name = "camera" + std::to_string(args.cameraIndex);
                } else if (!args.rtspUrl.empty()) {
                    run_name = "rtsp";
                } else {
                    run_name = fs::path(args.videoFile).stem().string();
                }
                run_dir = makeRunDir(args.saveDir, factory_->getModelName() + "_async",
                                     "stream", run_name);
                fs::create_directories(run_dir);
                writeRunInfo(run_dir, argv[0], args.modelPath, input_source_str);

                writer = initVideoWriter(
                    run_dir, fps > 0 ? fps : 30.0,
                    cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H),
                    video_save_path);
                if (!writer.isOpened()) {
                    std::cerr << "[ERROR] Failed to open video writer." << std::endl;
                    return -1;
                }
            }
        }

        std::cout << "[INFO] Starting async inference..." << std::endl;
        if (args.no_display) {
            std::cout << "Processing... Only FPS will be displayed." << std::endl;
        }

        cv::Mat display_image(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3);

        // Start display thread (render + save in background; imshow on main thread)
        std::thread displayThr([this, &visualizer, &args, &writer]() {
            displayThread(*visualizer, args.no_display, args.saveMode, writer);
        });

        // Set up async callback
        int buffer_index = 0;
        ie.RegisterCallback([this, postprocessor](
                            dxrt::TensorPtrs& outputs, void* user_data) -> int {  // NOSONAR(cpp:S5008)
            return onAsyncInferenceComplete(outputs, user_data, *postprocessor);
        });

        auto s_time = std::chrono::high_resolution_clock::now();
        int last_job_id = -1;
        int images_per_loop = is_image ? static_cast<int>(imageFiles.size()) : 0;

        if (is_image) {
            int effective_loop_count = (images_per_loop > 0) ? loopTest / images_per_loop : 1;
            for (int i = 0; i < loopTest && running_ && !g_interrupted(); ++i) {
                int loop_idx = i / images_per_loop;
                int img_idx  = i % images_per_loop;
                std::string currentImagePath = imageFiles[img_idx];

                // Loop banner
                if (effective_loop_count > 1 && img_idx == 0 && args.verbose) {
                    std::cout << "\n=================================================" << std::endl;
                    std::cout << "[INFO] Loop " << (loop_idx + 1) << "/" << effective_loop_count << std::endl;
                    std::cout << "=================================================" << std::endl;
                }
                if (args.verbose) {
                    std::cout << "\n[INFO] Image " << (img_idx + 1) << "/" << images_per_loop
                          << ": " << fs::path(currentImagePath).filename().string() << std::endl;
                }

                auto t_read_start = std::chrono::high_resolution_clock::now();
                cv::Mat img = cv::imread(currentImagePath);
                auto t_read_end = std::chrono::high_resolution_clock::now();

                if (img.empty()) {
                    std::cerr << "[ERROR] Failed to read image: " << currentImagePath << std::endl;
                    continue;
                }
                if (loop_idx == 0) {
                    if (args.verbose) {
                        std::cout << "[INFO] Input image: " << currentImagePath << std::endl;
                        std::cout << "[INFO] Image resolution (WxH): " << img.cols << "x" << img.rows << std::endl;
                    }
                }

                PreprocessContext ctx;
                cv::Mat preprocessed;
                // Preprocess original image so ctx.original_* are correct
                auto t_pre_start = std::chrono::high_resolution_clock::now();
                preprocessor->process(img, preprocessed, ctx);
                auto t_pre_end = std::chrono::high_resolution_clock::now();
                {
                    std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                    metrics_.sum_read += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
                    metrics_.sum_preprocess += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
                }
                // Prepare display image from original
                dxapp::displayResize(img, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);

                // Copy to pre-allocated buffer (convert to float if needed)
                auto& buf = input_buffers[buffer_index % ASYNC_BUFFER_SIZE];
                copyToInputBuffer(buf, preprocessed, is_float_input, is_nhwc);

                // Compute save_path for this image (dedup in loop mode)
                std::string save_path = computeImageSavePath(
                    image_save_dir, currentImagePath, images_per_loop, saved_files);

                // can render using original-pixel coordinates (not the resized display image).
                auto ud = std::make_unique<AsyncUserData>(AsyncUserData{img.clone(), ctx, std::move(save_path), {}});

                // Back-pressure: wait for available slot
                metrics_.waitForSlot();
                // Update inflight tracking
                updateInflightMetrics();

                ud->submit_ts = std::chrono::high_resolution_clock::now();
                last_job_id = ie.RunAsync(buf.data(), static_cast<void*>(ud.release()));
                buffer_index++;
                processCount++;
                if (!args.no_display) pollDisplay();
            }
        } else {
            // Video multi-loop
            for (int loop_idx = 0; loop_idx < loopTest && running_ && !g_interrupted(); ++loop_idx) {
                if (loopTest > 1 && args.verbose) {
                    std::cout << "\n=================================================" << std::endl;
                    std::cout << "[INFO] Loop " << (loop_idx + 1) << "/" << loopTest << std::endl;
                    std::cout << "=================================================" << std::endl;
                }

                auto readFrame = [&video](cv::Mat& f) { video >> f; return !f.empty(); };
                cv::Mat frame;
                while (running_ && !g_interrupted()) {
                    auto t_read_start = std::chrono::high_resolution_clock::now();
                    if (!readFrame(frame)) break;
                    auto t_read_end = std::chrono::high_resolution_clock::now();
                    PreprocessContext ctx;
                    cv::Mat preprocessed;
                    // Preprocess original video frame so ctx.original_* are correct
                    auto t_pre_start = std::chrono::high_resolution_clock::now();
                    preprocessor->process(frame, preprocessed, ctx);
                    auto t_pre_end = std::chrono::high_resolution_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                        metrics_.sum_read += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
                        metrics_.sum_preprocess += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
                    }
                    // Prepare display image from original frame
                    dxapp::displayResize(frame, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);

                    auto& buf = input_buffers[buffer_index % ASYNC_BUFFER_SIZE];
                    copyToInputBuffer(buf, preprocessed, is_float_input, is_nhwc);

                    // Use the original video frame (not the resized display image)
                    auto ud = std::make_unique<AsyncUserData>(AsyncUserData{frame.clone(), ctx, "", {}});
                    // Back-pressure: wait for available slot
                    metrics_.waitForSlot();
                    updateInflightMetrics();

                    ud->submit_ts = std::chrono::high_resolution_clock::now();
                    last_job_id = ie.RunAsync(buf.data(), static_cast<void*>(ud.release()));
                    buffer_index++;
                    processCount++;
                    if (!args.no_display) pollDisplay();
                }

                // Reopen video for next loop iteration
                if (loop_idx + 1 >= loopTest || args.videoFile.empty()) continue;
                video.release();
                video.open(args.videoFile);
                if (!video.isOpened()) {
                    std::cerr << "[ERROR] Failed to reopen video for loop " << (loop_idx + 2) << std::endl;
                    break;
                }
            }
        }

        // Wait for inference completion while keeping display responsive
        if (last_job_id >= 0) {
            if (!running_) display_queue_.shutdown();
            std::atomic<bool> inference_done{false};
            std::thread waitThread([&ie, last_job_id, &inference_done]() {
                ie.Wait(last_job_id);
                inference_done.store(true, std::memory_order_release);
            });
            while (!inference_done.load(std::memory_order_acquire) && running_) {
                if (!args.no_display) pollDisplay();
                else std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            waitThread.join();
        }
        // For images: keep display alive until user closes window
        if (is_image && !args.no_display) {
            // Drain remaining rendered frames
            while (running_) {
                if (!pollDisplay()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        running_ = false;
        display_queue_.shutdown();
        rendered_queue_.shutdown();
        displayThr.join();
        cv::destroyAllWindows();

        auto e_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(e_time - s_time).count();

        if (g_interrupted()) {
            std::cout << "\n[INFO] Interrupted by user (Ctrl+C)" << std::endl;
        }

        if (writer.isOpened()) {
            writer.release();
            if (!video_save_path.empty()) {
                if (args.verbose) {
                    std::cout << "\n[INFO] Saved output video: " << fs::absolute(video_save_path).string() << std::endl;
                }
            }
        }

        printPerformanceSummary(processCount, total_time, !args.no_display, args.saveMode);

        DXRT_TRY_CATCH_END
        return 0;
    }

private:
    std::unique_ptr<FactoryT> factory_;
    std::string model_path_;
    std::atomic<bool> running_{true};
    bool window_shown_ = false;
    bool window_prop_supported_ = true;  // false if backend always returns -1
    SafeQueue<AsyncDisplayArgs> display_queue_;
    SafeQueue<cv::Mat> rendered_queue_;  // Rendered frames for main-thread display
    AsyncProfilingMetrics metrics_;

    /** Handle async inference completion: postprocess, log, update metrics, enqueue display. */
    int onAsyncInferenceComplete(
        dxrt::TensorPtrs& outputs, void* user_data,  // NOSONAR(cpp:S5008)
        IPostprocessor<DetectionResult>& postprocessor) {
        auto* ud = static_cast<AsyncUserData*>(user_data);
        auto t_callback_start = std::chrono::high_resolution_clock::now();

        // Measure inference time (submit → callback)
        {
            std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
            double t_inf = std::chrono::duration<double, std::milli>(t_callback_start - ud->submit_ts).count();
            metrics_.sum_inference += t_inf;
        }

        auto t_post_start = std::chrono::high_resolution_clock::now();

        std::vector<DetectionResult> detections;
        try {
            detections = postprocessor.process(outputs, ud->ctx);
        } catch (const std::exception& e) {
            std::cerr << "[DXAPP] [ER] Postprocess error: " << e.what() << std::endl;
        }

        printDetectionResults(detections, ud->display_frame.cols, ud->display_frame.rows, verbose_);

        auto t_post_end = std::chrono::high_resolution_clock::now();
        double t_postprocess = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();
            { std::lock_guard<std::mutex> lock(metrics_.metrics_mutex); metrics_.sum_postprocess += t_postprocess; }

        completeInflightMetrics();

        AsyncDisplayArgs display_args;
        display_args.original_frame = std::make_shared<cv::Mat>(ud->display_frame);
        display_args.detections = std::make_shared<std::vector<DetectionResult>>(std::move(detections));
        display_args.t_postprocess = t_postprocess;
        display_args.ctx = ud->ctx;
        display_args.save_path = std::move(ud->save_path);

        verify::dumpVerifyJson(detections, model_path_,
            "object_detection", ud->display_frame.rows, ud->display_frame.cols);

        display_queue_.push(std::move(display_args));
        auto guard = std::unique_ptr<AsyncUserData>{ud};
        (void)guard;
        return 0;
    }

    /** Print detection results to stdout for pipeline parsing. */
    static void printDetectionResults(const std::vector<DetectionResult>& detections,
                                      int frame_width, int frame_height, bool verbose = false) {
        for (size_t di = 0; di < detections.size(); ++di) {
            const auto& det = detections[di];
            if (det.box.size() >= 4) {
                std::string dname = dxapp::sanitize_name(det.class_name);
                if (verbose) {
                    std::cout << "[DET] " << dname
                              << " " << det.confidence
                              << " " << det.box[0] << " " << det.box[1]
                              << " " << det.box[2] << " " << det.box[3]
                              << " " << frame_width << " " << frame_height
                              << std::endl;
                }
            }
        }
    }

    /** Update metrics when an async inference job completes. */
    void completeInflightMetrics() {
        {
            std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
            auto now = std::chrono::high_resolution_clock::now();
            metrics_.inflight_current--;
            if (!metrics_.first_inference) {
                auto elapsed = std::chrono::duration<double>(now - metrics_.inflight_last_ts).count();
                metrics_.inflight_time_sum += metrics_.inflight_current * elapsed;
            }
            metrics_.inflight_last_ts = now;
            metrics_.infer_last_ts = now;
            metrics_.infer_completed++;
        }
        metrics_.notifySlot();
    }

    /** Copy preprocessed data to async input buffer (float conversion if needed). */
    static void copyToInputBuffer(std::vector<uint8_t>& buf, const cv::Mat& preprocessed,
                                   bool is_float_input, bool is_nhwc) {
        if (is_float_input && !preprocessed.empty()) {
            auto float_data = convertToFloatBuffer(preprocessed, is_nhwc);
            std::memcpy(buf.data(), float_data.data(), float_data.size() * sizeof(float));
        } else {
            std::memcpy(buf.data(), preprocessed.data,
                        preprocessed.total() * preprocessed.elemSize());
        }
    }

    /** Update in-flight tracking metrics (single producer/consumer). */
    void updateInflightMetrics() {
        std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
        auto now = std::chrono::high_resolution_clock::now();
        if (metrics_.first_inference) {
            metrics_.infer_first_ts = now;
            metrics_.inflight_last_ts = now;
            metrics_.first_inference = false;
        } else {
            auto elapsed = std::chrono::duration<double>(now - metrics_.inflight_last_ts).count();
            metrics_.inflight_time_sum += metrics_.inflight_current * elapsed;
            metrics_.inflight_last_ts = now;
        }
        metrics_.inflight_current++;
        if (metrics_.inflight_current > metrics_.inflight_max) {
            metrics_.inflight_max = metrics_.inflight_current;
        }
    }

    /** Compute deduplicated save path for an image frame. */
    static std::string computeImageSavePath(
        const std::string& image_save_dir, const std::string& imagePath,
        int images_per_loop, std::set<std::string, std::less<>>& saved_files) {
        if (image_save_dir.empty()) return "";
        std::string base_filename = fs::path(imagePath).filename().string();
        std::string potential_path = (images_per_loop > 1)
            ? (image_save_dir + "/" + base_filename + "/output.jpg")
            : (image_save_dir + "/output.jpg");
        if (images_per_loop > 1) {
            fs::create_directories(image_save_dir + "/" + base_filename);
        }
        if (saved_files.find(potential_path) != saved_files.end()) return "";
        saved_files.insert(potential_path);
        return potential_path;
    }

    // --- Command line parsing (matching Legacy format) ---

    CommandLineArgs parseCommandLine(int argc, char* argv[]) {
        CommandLineArgs args;
        std::string app_name = factory_->getModelName() + " Post-Processing Async Example";
        cxxopts::Options options(app_name, app_name + " application usage ");
        options.add_options()
            ("m, model_path", "object detection model file (.dxnn, required)",
             cxxopts::value<std::string>(args.modelPath))
            ("i, image_path", "input image file path or directory containing images (supports jpg, png, jpeg, bmp)",
             cxxopts::value<std::string>(args.imageFilePath))
            ("v, video_path", "input video file path(mp4, mov, avi ...)",
             cxxopts::value<std::string>(args.videoFile))
            ("c, camera_index", "camera device index (e.g., 0)",
             cxxopts::value<int>(args.cameraIndex))
            ("r, rtsp_url", "RTSP stream URL",
             cxxopts::value<std::string>(args.rtspUrl))
            ("s, save", "Save rendered output to disk",
             cxxopts::value<bool>(args.saveMode)->default_value("false"))
            ("save-dir", "Base directory for run outputs when using --save/--dump-tensors.",
             cxxopts::value<std::string>(args.saveDir)->default_value("artifacts/cpp_example"))
            ("dump-tensors", "(Debug) Always dump input/output tensors as .bin files.",
             cxxopts::value<bool>(args.dumpTensors)->default_value("false"))
            ("l, loop", "Number of inference iterations to run",
             cxxopts::value<int>(args.loopTest)->default_value("-1"))
            ("no-display", "will not visualize, only show fps",
             cxxopts::value<bool>(args.no_display)->default_value("false"))
            ("config", "Model configuration JSON file path",
             cxxopts::value<std::string>(args.configPath))
            ("show-log", "Enable verbose log output (default: quiet)",
             cxxopts::value<bool>(args.verbose)->default_value("false"))
            ("h, help", "print usage");

        auto cmd = options.parse(argc, argv);
        if (cmd.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        return args;
    }

    void validateArguments(const CommandLineArgs& args) {
        if (args.modelPath.empty()) {
            dxapp::fatal_error("[ERROR] Model path is required. Use -m or --model_path option.\nUse -h or --help for usage information.");
        }

        int sourceCount = 0;
        if (!args.imageFilePath.empty()) sourceCount++;
        if (!args.videoFile.empty()) sourceCount++;
        if (args.cameraIndex >= 0) sourceCount++;
        if (!args.rtspUrl.empty()) sourceCount++;

        if (sourceCount != 1) {
            dxapp::fatal_error("[ERROR] Please specify exactly one input source: image (-i), video (-v), "
                              "camera (-c), or RTSP (-r).\nUse -h or --help for usage information.");
        }
    }

    std::pair<std::vector<std::string>, int> processImagePath(
        const std::string& imageFilePath, int loopTest) {
        std::vector<std::string> imageFiles;

        if (fs::is_directory(imageFilePath)) {
            for (const auto& entry : fs::directory_iterator(imageFilePath)) {
                if (!fs::is_regular_file(entry.path())) continue;
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    imageFiles.push_back(entry.path().string());
                }
            }
            std::sort(imageFiles.begin(), imageFiles.end());
            if (imageFiles.empty()) {
                dxapp::fatal_error("[ERROR] No image files found in directory: ");
            }
            if (loopTest == -1) {
                loopTest = static_cast<int>(imageFiles.size());
            }
        } else if (fs::is_regular_file(imageFilePath)) {
            imageFiles.push_back(imageFilePath);
            if (loopTest == -1) loopTest = 1;
        } else {
            dxapp::fatal_error("[ERROR] Invalid image path: ");
        }

        return {imageFiles, loopTest};
    }

    bool openVideoCapture(cv::VideoCapture& video, const CommandLineArgs& args) {
        if (args.cameraIndex >= 0) {
            video.open(args.cameraIndex);
        } else if (!args.rtspUrl.empty()) {
            video.open(args.rtspUrl);
        } else {
            video.open(args.videoFile);
        }
        return video.isOpened();
    }

    void displayThread(IVisualizer<DetectionResult>& visualizer, bool no_display,
                       bool save_on, cv::VideoWriter& writer) {
        while (running_) {
            AsyncDisplayArgs args;
            if (!display_queue_.try_pop(args, std::chrono::milliseconds(100))) {
                continue;
            }

            if (!args.original_frame || args.original_frame->empty()) {
                continue;
            }

            // Render
            auto t_render_start = std::chrono::high_resolution_clock::now();
            cv::Mat result_frame = args.original_frame->clone();
            if (args.detections) {
                result_frame = visualizer.draw(result_frame, *args.detections, args.ctx);
            }
            auto t_render_end = std::chrono::high_resolution_clock::now();
            double render_time = std::chrono::duration<double, std::milli>(
                t_render_end - t_render_start).count();

            // Save
            auto t_save_start = std::chrono::high_resolution_clock::now();
            if (!args.save_path.empty() && !result_frame.empty()) {
                cv::imwrite(args.save_path, result_frame);
                if (verbose_) {
                    std::cout << "\n[INFO] Saved output image: " << fs::absolute(args.save_path).string() << std::endl;
                }
            }
            if (save_on && writer.isOpened() && !result_frame.empty()) {
                writer << result_frame;
            }
            dxapp::saveDebugImage(result_frame);
            auto t_save_end = std::chrono::high_resolution_clock::now();
            double save_time = std::chrono::duration<double, std::milli>(
                t_save_end - t_save_start).count();

            // Update metrics
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                metrics_.sum_render += render_time;
                metrics_.sum_save += save_time;
            }

            // Push rendered frame for main-thread display (imshow must run on main thread for Qt)
            if (!no_display && !result_frame.empty()) {
                rendered_queue_.push(result_frame.clone());
            }
        }
    }

    /** Poll rendered_queue_ and display on main thread. Returns false if user requested quit. */
    bool pollDisplay() {
        cv::Mat frame;
        if (rendered_queue_.try_pop(frame, std::chrono::milliseconds(1))) {
            cv::imshow("Output", frame);
            if (!window_shown_) {
                window_shown_ = true;
                // Probe backend: some backends (e.g. GTK2) always return -1
                // for WND_PROP_VISIBLE. Detect this on the first frame so we
                // never falsely interpret -1 as "window closed by user".
                cv::waitKey(1);
                double probe = cv::getWindowProperty("Output", cv::WND_PROP_VISIBLE);
                if (probe < 0.0) {
                    window_prop_supported_ = false;
                }
                return true;
            }
        }
        if (!window_shown_) return true;  // window not created yet, skip checks
        // Pump events and detect user quit / window close
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            running_ = false;
            display_queue_.shutdown();
            rendered_queue_.shutdown();
            return false;
        }
        if (window_prop_supported_) {
            double vis = cv::getWindowProperty("Output", cv::WND_PROP_VISIBLE);
            if (vis < 0.0) {
                running_ = false;
                display_queue_.shutdown();
                rendered_queue_.shutdown();
                return false;
            }
        }
        return true;
    }

    // --- Performance summary (matching Legacy async format exactly) ---

    void printPerformanceSummary(int total_frames, double total_time_sec, bool /*display_on*/, bool save_on = false) {
        std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
        if (metrics_.infer_completed == 0) return;
        double avg_read = metrics_.sum_read / metrics_.infer_completed;
        double avg_pre = metrics_.sum_preprocess / metrics_.infer_completed;
        double avg_inf = metrics_.sum_inference / metrics_.infer_completed;
        double avg_post = metrics_.sum_postprocess / metrics_.infer_completed;
        auto inflight_time_window = std::chrono::duration<double>(metrics_.infer_last_ts - metrics_.infer_first_ts).count();
        double infer_tp = (inflight_time_window > 0) ? metrics_.infer_completed / inflight_time_window : 0.0;
        double inflight_avg = (inflight_time_window > 0) ? metrics_.inflight_time_sum / inflight_time_window : 0.0;
        auto printRow = [&](const char* name, double avg_ms, double fps, const char* suffix = "") {
            std::cout << " " << std::left << std::setw(15) << name << std::right << std::setw(8)
                      << std::fixed << std::setprecision(2) << avg_ms << " ms     " << std::setw(6)
                      << std::setprecision(1) << fps << " FPS" << suffix << std::endl;
        };
        std::cout << "\n==================================================" << std::endl;
        std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        printRow("Read", avg_read, avg_read > 0 ? 1000.0/avg_read : 0.0);
        printRow("Preprocess", avg_pre, avg_pre > 0 ? 1000.0/avg_pre : 0.0);
        printRow("Inference", avg_inf, infer_tp, "*");
        printRow("Postprocess", avg_post, avg_post > 0 ? 1000.0/avg_post : 0.0);
        if (save_on && metrics_.sum_save > 0) {
            double avg_save = metrics_.sum_save / metrics_.infer_completed;
            printRow("Save", avg_save, avg_save > 0 ? 1000.0/avg_save : 0.0);
        }
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << " * Async: turnaround latency (submit to callback)" << std::endl;
        std::cout << "   Throughput measured independently" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << " " << std::left << std::setw(19) << "Infer Completed" << " :    " << metrics_.infer_completed << std::endl;
        std::cout << " " << std::left << std::setw(19) << "Infer Inflight Avg" << " :    " << std::fixed << std::setprecision(1) << inflight_avg << std::endl;
        std::cout << " " << std::left << std::setw(19) << "Infer Inflight Max" << " :      " << metrics_.inflight_max << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << " " << std::left << std::setw(19) << "Total Frames" << " :    " << total_frames << std::endl;
        std::cout << " " << std::left << std::setw(19) << "Total Time" << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s" << std::endl;
        double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;
        std::cout << " " << std::left << std::setw(19) << "Overall FPS" << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS" << std::endl;
        std::cout << "==================================================" << std::endl;
    }
};

}  // namespace dxapp

#endif  // ASYNC_RUNNER_HPP
