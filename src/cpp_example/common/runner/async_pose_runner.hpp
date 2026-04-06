/**
 * @file async_pose_runner.hpp
 * @brief Asynchronous pose estimation runner using factory pattern
 *
 * Provides a generic async runner that accepts any IPoseFactory implementation.
 */

#ifndef ASYNC_POSE_RUNNER_HPP
#define ASYNC_POSE_RUNNER_HPP

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
#include <thread>
#include <vector>

#include "common/base/i_factory.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/verify_serialize.hpp"
#include "async_detection_runner.hpp"

namespace dxapp {

struct AsyncPoseDisplayArgs {
    std::shared_ptr<std::vector<PoseResult>> detections;
    std::shared_ptr<cv::Mat> original_frame;
    std::string save_path;
    double t_read = 0.0;
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    PreprocessContext ctx;
    AsyncPoseDisplayArgs() = default;
    AsyncPoseDisplayArgs(const AsyncPoseDisplayArgs&) = default;
    AsyncPoseDisplayArgs& operator=(const AsyncPoseDisplayArgs&) = default;
    AsyncPoseDisplayArgs(AsyncPoseDisplayArgs&&) noexcept = default;
    AsyncPoseDisplayArgs& operator=(AsyncPoseDisplayArgs&&) noexcept = default;
};

template <typename FactoryT>
class AsyncPoseRunner {
    bool verbose_ = false;

public:
    explicit AsyncPoseRunner(std::unique_ptr<FactoryT> factory)
        : factory_(std::move(factory)) {}

    int run(int argc, char* argv[]) {
        DXRT_TRY_CATCH_BEGIN
        installSignalHandlers();
        int processCount = 0;

        CommandLineArgs args = parseCommandLine(argc, argv);
        verbose_ = args.verbose;
        validateArguments(args);

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

        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(args.modelPath, io);
        model_path_ = args.modelPath;

        if (!dxapp::minversionforRTandCompiler(&ie)) {
            std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                         "compatible with the version of the runtime. Please compile the model again." << std::endl;
            return -1;
        }

        auto input_shape = ie.GetInputs().front().shape();
        int input_height, input_width;
        parseInputShape(input_shape, input_width, input_height);

        // Load model configuration if provided
        if (!args.configPath.empty()) {
            dxapp::ModelConfig config(args.configPath);
            factory_->loadConfig(config);
        }

        auto preprocessor = factory_->createPreprocessor(input_width, input_height);
        // Use shared_ptr so the callback lambda can capture by value and extend lifetime
        // past ie.Wait() which may return before the background callback thread completes.
        auto postprocessor_uptr = factory_->createPostprocessor(input_width, input_height, ie.IsOrtConfigured());
        auto postprocessor = std::shared_ptr<typename decltype(postprocessor_uptr)::element_type>(std::move(postprocessor_uptr));
        auto visualizer = factory_->createVisualizer();

        std::cout << "[INFO] Model loaded: " << args.modelPath << std::endl;
        std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height << std::endl;
        std::cout << std::endl;

        size_t input_size = ie.GetInputSize();
        std::vector<std::vector<uint8_t>> input_buffers(ASYNC_BUFFER_SIZE, std::vector<uint8_t>(input_size));

        cv::VideoCapture video;
        cv::VideoWriter writer;
        std::string run_dir;
        std::string video_save_path;

        if (!is_image) {
            if (!openVideoCapture(video, args)) {
                std::cerr << "[ERROR] Failed to open input source." << std::endl;
                return -1;
            }
            auto frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
            auto frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps = video.get(cv::CAP_PROP_FPS);
            auto total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
            std::string source_info;
            if (args.cameraIndex >= 0) source_info = "Camera index: " + std::to_string(args.cameraIndex);
            else if (!args.rtspUrl.empty()) source_info = "RTSP URL: " + args.rtspUrl;
            else { source_info = "Video file: " + args.videoFile; }
            if (args.verbose) {
                std::cout << "[INFO] " << source_info << std::endl;
                std::cout << "[INFO] Input source resolution (WxH): " << frame_width << "x" << frame_height << std::endl;
                std::cout << "[INFO] Input source FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
                if (!args.videoFile.empty()) std::cout << "[INFO] Total frames: " << total_frames << std::endl;
                std::cout << std::endl;
            }
            if (args.saveMode) {
                std::string run_name;
                if (args.cameraIndex >= 0) run_name = "camera" + std::to_string(args.cameraIndex);
                else if (!args.rtspUrl.empty()) run_name = "rtsp";
                else run_name = fs::path(args.videoFile).stem().string();
                std::string input_src = buildInputSourceString(
                    args.imageFilePath, args.videoFile, args.cameraIndex, args.rtspUrl);
                run_dir = makeRunDir(args.saveDir, factory_->getModelName() + "_async",
                                     "stream", run_name);
                fs::create_directories(run_dir);
                writeRunInfo(run_dir, argv[0], args.modelPath, input_src);
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
        if (args.no_display) std::cout << "Processing... Only FPS will be displayed." << std::endl;

        cv::Mat display_image(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3);

        std::thread displayThr([this, &visualizer, &args, &writer]() {
            displayThread(*visualizer, args.no_display, args.saveMode, writer);
        });

        int buffer_index = 0;
        ie.RegisterCallback([this, postprocessor](dxrt::TensorPtrs& outputs, void* user_data) -> int {  // NOSONAR(cpp:S5008)
            if (!user_data) return 0;
            auto* ud = static_cast<AsyncUserData*>(user_data);
            auto t_post_start = std::chrono::high_resolution_clock::now();

            std::vector<PoseResult> detections;
            try { detections = postprocessor->process(outputs, ud->ctx); }
            catch (const std::exception& e) { std::cerr << "[DXAPP] [ER] Postprocess error: " << e.what() << std::endl; }

            auto t_post_end = std::chrono::high_resolution_clock::now();
            double t_postprocess = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();
            { std::lock_guard<std::mutex> lock(metrics_.metrics_mutex); metrics_.sum_postprocess += t_postprocess; }

            // Print pose estimation results for pipeline parsing
            if (!detections.empty() && verbose_) {
                std::cout << "[POSE] " << detections.size();
                for (const auto& p : detections) {
                    std::cout << " " << std::fixed << std::setprecision(2) << p.confidence;
                }
                std::cout << std::endl;
            }

            auto now = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                // Measure inference time (submit → callback)
                double t_inf = std::chrono::duration<double, std::milli>(now - ud->submit_ts).count();
                metrics_.sum_inference += t_inf;
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

            AsyncPoseDisplayArgs display_args;
            display_args.original_frame = std::make_shared<cv::Mat>(ud->display_frame);
            display_args.detections = std::make_shared<std::vector<PoseResult>>(std::move(detections));
            display_args.t_postprocess = t_postprocess;
            display_args.ctx = ud->ctx;
            display_args.save_path = std::move(ud->save_path);
            // --- Numerical verification dump (DXAPP_VERIFY=1) ---
            verify::dumpVerifyJson(*display_args.detections, model_path_,
                "pose_estimation", display_args.original_frame->rows, display_args.original_frame->cols);

            display_queue_.push(std::move(display_args));
            delete ud;
            return 0;
        });

        auto s_time = std::chrono::high_resolution_clock::now();
        int last_job_id = -1;

        auto processFrameAsync = [&](const cv::Mat& frame) {
            auto t_pre_start = std::chrono::high_resolution_clock::now();
            dxapp::displayResize(frame, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
            PreprocessContext ctx;
            cv::Mat preprocessed;
            preprocessor->process(display_image, preprocessed, ctx);
            auto t_pre_end = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                metrics_.sum_preprocess += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
            }
            auto& buf = input_buffers[buffer_index % ASYNC_BUFFER_SIZE];
            std::memcpy(buf.data(), preprocessed.data, preprocessed.total() * preprocessed.elemSize());
            auto user_data_ptr = std::make_unique<AsyncUserData>(AsyncUserData{display_image.clone(), ctx, std::string(), {}});
            void* user_data = user_data_ptr.release();
            metrics_.waitForSlot();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                auto now = std::chrono::high_resolution_clock::now();
                if (metrics_.first_inference) {
                    metrics_.infer_first_ts = now; metrics_.inflight_last_ts = now; metrics_.first_inference = false;
                } else {
                    auto elapsed = std::chrono::duration<double>(now - metrics_.inflight_last_ts).count();
                    metrics_.inflight_time_sum += metrics_.inflight_current * elapsed;
                    metrics_.inflight_last_ts = now;
                }
                metrics_.inflight_current++;
                if (metrics_.inflight_current > metrics_.inflight_max) metrics_.inflight_max = metrics_.inflight_current;
            }
            static_cast<AsyncUserData*>(user_data)->submit_ts = std::chrono::high_resolution_clock::now();
            last_job_id = ie.RunAsync(buf.data(), user_data);
            buffer_index++;
            processCount++;
            if (!args.no_display) pollDisplay();
        };

        if (is_image) {
            // Prepare run dir for image saves
            if (args.saveMode) {
                std::string run_kind = fs::is_directory(args.imageFilePath) ? "image-dir" : "image";
                std::string run_name = fs::path(args.imageFilePath).filename().string();
                std::string input_src = buildInputSourceString(
                    args.imageFilePath, args.videoFile, args.cameraIndex, args.rtspUrl);
                run_dir = makeRunDir(args.saveDir, factory_->getModelName() + "_async",
                                     run_kind, run_name);
                fs::create_directories(run_dir);
                writeRunInfo(run_dir, argv[0], args.modelPath, input_src);
            }

            for (int i = 0; i < loopTest && running_ && !g_interrupted(); ++i) {
                auto t_read_start = std::chrono::high_resolution_clock::now();
                cv::Mat img = cv::imread(imageFiles[i % imageFiles.size()]);
                auto t_read_end = std::chrono::high_resolution_clock::now();
                if (img.empty()) continue;
                dxapp::displayResize(img, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
                PreprocessContext ctx;
                cv::Mat preprocessed;
                auto t_pre_start = std::chrono::high_resolution_clock::now();
                preprocessor->process(display_image, preprocessed, ctx);
                auto t_pre_end = std::chrono::high_resolution_clock::now();
                {
                    std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                    metrics_.sum_read += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
                    metrics_.sum_preprocess += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
                }
                auto& buf = input_buffers[buffer_index % ASYNC_BUFFER_SIZE];
                std::memcpy(buf.data(), preprocessed.data, preprocessed.total() * preprocessed.elemSize());
                std::string save_path;
                if (!run_dir.empty()) {
                    save_path = dxapp::buildPerImageSavePath(run_dir, factory_->getModelName() + "_async", imageFiles[i % imageFiles.size()], i);
                }
                auto ud = std::make_unique<AsyncUserData>(AsyncUserData{display_image.clone(), ctx, std::move(save_path), {}});
                metrics_.waitForSlot();
                {
                    std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                    auto now = std::chrono::high_resolution_clock::now();
                    if (metrics_.first_inference) {
                        metrics_.infer_first_ts = now; metrics_.inflight_last_ts = now; metrics_.first_inference = false;
                    } else {
                        auto elapsed = std::chrono::duration<double>(now - metrics_.inflight_last_ts).count();
                        metrics_.inflight_time_sum += metrics_.inflight_current * elapsed;
                        metrics_.inflight_last_ts = now;
                    }
                    metrics_.inflight_current++;
                    if (metrics_.inflight_current > metrics_.inflight_max) metrics_.inflight_max = metrics_.inflight_current;
                }
                ud->submit_ts = std::chrono::high_resolution_clock::now();
                last_job_id = ie.RunAsync(buf.data(), static_cast<void*>(ud.release()));
                buffer_index++;
                processCount++;
                if (!args.no_display) pollDisplay();
            }
        } else {
            for (int loop_idx = 0; loop_idx < loopTest && running_ && !g_interrupted(); ++loop_idx) {
                if (loopTest > 1) {
                    if (args.verbose) {
                        std::cout << "\n" << std::string(50, '=') << std::endl;
                        std::cout << "[INFO] Loop " << (loop_idx + 1) << "/" << loopTest << std::endl;
                        std::cout << std::string(50, '=') << std::endl;
                    }
                }
                auto readFrame = [&video](cv::Mat& f) { video >> f; return !f.empty(); };
                cv::Mat frame;
                while (running_ && !g_interrupted()) {
                    auto t_read_start = std::chrono::high_resolution_clock::now();
                    if (!readFrame(frame)) break;
                    auto t_read_end = std::chrono::high_resolution_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                        metrics_.sum_read += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
                    }
                    processFrameAsync(frame);
                }
                // Reopen video for next loop
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
    SafeQueue<AsyncPoseDisplayArgs> display_queue_;
    SafeQueue<cv::Mat> rendered_queue_;  // Rendered frames for main-thread display
    AsyncProfilingMetrics metrics_;

    CommandLineArgs parseCommandLine(int argc, char* argv[]) {
        CommandLineArgs args;
        std::string app_name = factory_->getModelName() + " Post-Processing Async Example";
        cxxopts::Options options(app_name, app_name + " application usage ");
        options.add_options()
            ("m, model_path", "model file (.dxnn, required)", cxxopts::value<std::string>(args.modelPath))
            ("i, image_path", "input image file path or directory", cxxopts::value<std::string>(args.imageFilePath))
            ("v, video_path", "input video file path", cxxopts::value<std::string>(args.videoFile))
            ("c, camera_index", "camera device index", cxxopts::value<int>(args.cameraIndex))
            ("r, rtsp_url", "RTSP stream URL", cxxopts::value<std::string>(args.rtspUrl))
            ("s, save", "Save rendered output to disk", cxxopts::value<bool>(args.saveMode)->default_value("false"))
            ("save-dir", "Base directory for run outputs when using --save/--dump-tensors.", cxxopts::value<std::string>(args.saveDir)->default_value("artifacts/cpp_example"))
            ("dump-tensors", "(Debug) Always dump input/output tensors as .bin files.", cxxopts::value<bool>(args.dumpTensors)->default_value("false"))
            ("l, loop", "Number of inference iterations", cxxopts::value<int>(args.loopTest)->default_value("-1"))
            ("no-display", "will not visualize, only show fps", cxxopts::value<bool>(args.no_display)->default_value("false"))
            ("config", "Model configuration JSON file path",
             cxxopts::value<std::string>(args.configPath))
            ("show-log", "Enable verbose log output (default: quiet)",
             cxxopts::value<bool>(args.verbose)->default_value("false"))
            ("h, help", "print usage");
        auto cmd = options.parse(argc, argv);
        if (cmd.count("help")) { std::cout << options.help() << std::endl; exit(0); }
        return args;
    }

    void validateArguments(const CommandLineArgs& args) {
        if (args.modelPath.empty()) { dxapp::fatal_error("[ERROR] Model path is required."); }
        int sourceCount = 0;
        if (!args.imageFilePath.empty()) sourceCount++;
        if (!args.videoFile.empty()) sourceCount++;
        if (args.cameraIndex >= 0) sourceCount++;
        if (!args.rtspUrl.empty()) sourceCount++;
        if (sourceCount != 1) { dxapp::fatal_error("[ERROR] Please specify exactly one input source."); }
    }

    std::pair<std::vector<std::string>, int> processImagePath(const std::string& imageFilePath, int loopTest) {
        std::vector<std::string> imageFiles;
        if (fs::is_directory(imageFilePath)) {
            for (const auto& entry : fs::directory_iterator(imageFilePath)) {
                if (!fs::is_regular_file(entry.path())) continue;
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                    imageFiles.push_back(entry.path().string());
            }
            std::sort(imageFiles.begin(), imageFiles.end());
            if (imageFiles.empty()) { dxapp::fatal_error("[ERROR] No image files found."); }
            if (loopTest == -1) loopTest = static_cast<int>(imageFiles.size());
        } else if (fs::is_regular_file(imageFilePath)) {
            imageFiles.push_back(imageFilePath);
            if (loopTest == -1) loopTest = 1;
        } else { dxapp::fatal_error("[ERROR] Invalid image path."); }
        return {imageFiles, loopTest};
    }

    bool openVideoCapture(cv::VideoCapture& video, const CommandLineArgs& args) {
        if (args.cameraIndex >= 0) video.open(args.cameraIndex);
        else if (!args.rtspUrl.empty()) video.open(args.rtspUrl);
        else video.open(args.videoFile);
        return video.isOpened();
    }

    void displayThread(IVisualizer<PoseResult>& visualizer, bool no_display,
                       bool save_on, cv::VideoWriter& writer) {
        while (running_) {
            AsyncPoseDisplayArgs args;
            if (!display_queue_.try_pop(args, std::chrono::milliseconds(100))) continue;
            if (!args.original_frame || args.original_frame->empty()) continue;
            auto t_render_start = std::chrono::high_resolution_clock::now();
            cv::Mat result_frame = args.original_frame->clone();
            if (args.detections) result_frame = visualizer.draw(result_frame, *args.detections, args.ctx);
            auto t_render_end = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                metrics_.sum_render += std::chrono::duration<double, std::milli>(t_render_end - t_render_start).count();
            }
            if (!args.save_path.empty() && !result_frame.empty()) {
                cv::imwrite(args.save_path, result_frame);
                if (verbose_) {
                    std::cout << "\n[INFO] Saved output image: " << fs::absolute(args.save_path).string() << std::endl;
                }
            }
            if (save_on && writer.isOpened() && !result_frame.empty()) writer << result_frame;
            dxapp::saveDebugImage(result_frame);
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

#endif  // ASYNC_POSE_RUNNER_HPP
