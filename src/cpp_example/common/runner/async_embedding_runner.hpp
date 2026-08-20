/**
 * @file async_embedding_runner.hpp
 * @brief Asynchronous embedding/feature extraction runner using factory pattern
 *
 * Provides a generic async runner that accepts any IEmbeddingFactory implementation.
 */

#ifndef ASYNC_EMBEDDING_RUNNER_HPP
#define ASYNC_EMBEDDING_RUNNER_HPP

#include <dxrt/dxrt_api.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cxxopts.hpp>
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
#include "common/utility/frame_reorder.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/verify_serialize.hpp"
#include "async_detection_runner.hpp"

namespace dxapp {

struct AsyncEmbeddingDisplayArgs {
    std::shared_ptr<std::vector<EmbeddingResult>> results;
    std::shared_ptr<cv::Mat> original_frame;
    std::string save_path;
    double t_read = 0.0;
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    PreprocessContext ctx;
    uint64_t frame_index = 0;  // Monotonic submit order, for in-order display
    AsyncEmbeddingDisplayArgs() = default;
    AsyncEmbeddingDisplayArgs(const AsyncEmbeddingDisplayArgs&) = default;
    AsyncEmbeddingDisplayArgs& operator=(const AsyncEmbeddingDisplayArgs&) = default;
    AsyncEmbeddingDisplayArgs(AsyncEmbeddingDisplayArgs&&) noexcept = default;
    AsyncEmbeddingDisplayArgs& operator=(AsyncEmbeddingDisplayArgs&&) noexcept = default;
};

template <typename FactoryT>
class AsyncEmbeddingRunner {
    bool verbose_ = false;

public:
    explicit AsyncEmbeddingRunner(std::unique_ptr<FactoryT> factory)
        : factory_(std::move(factory)) {}

    int run(int argc, char* argv[]) {
        DXRT_TRY_CATCH_BEGIN
        installSignalHandlers();
        int processCount = 0;

        CommandLineArgs args = parseCommandLine(argc, argv);
        verbose_ = args.verbose;

        if (verbose_) {
            std::cout << "[DXAPP] [INFO] --show-log: This task produces image-based output. "
                         "Use --save or display mode to view results." << std::endl;
        }
        // Image-only tasks: if no input was given, show a hint and exit normally
        // (do not auto-run inference on a default sample).
        std::string task = factory_->getTaskType();
        bool hasStreamInput = !args.videoFile.empty() || args.cameraIndex >= 0 || !args.rtspUrl.empty();
        if ((task == "embedding" || task == "reid" || task == "attribute_recognition") && hasStreamInput) {
            dxapp::fatal_error("[DXAPP] [ERROR] Task '" + task + "' supports image input only (-i / --image_path). "
                "Video/camera input requires a detection crop pipeline and is not supported in single-model examples. "
                "Use -i (--image_path) to provide an image file or directory.");
        }
        if ((task == "embedding" || task == "reid" || task == "attribute_recognition")
            && args.imageFilePath.empty()) {
            std::cout << "[DXAPP] [INFO] Task '" << task << "' takes image input only." << std::endl;
            std::cout << "        -> Provide an image with -i (--image_path), e.g. -i "
                      << dxapp::getDefaultSampleImage(task) << std::endl;
            return 0;
        }
        // Apply default sample image if no input specified
        if (args.imageFilePath.empty()) {
            args.imageFilePath = dxapp::getDefaultSampleImage(factory_->getTaskType());
            std::cout << "[DXAPP] [INFO] No input specified. Using default sample: " << args.imageFilePath << std::endl;
        }
        dxapp::resolveAndValidateModel(args.modelPath, argv[0]);
        validateArguments(args);

        std::vector<std::string> imageFiles;
        is_image_mode_ = true;
        int loopTest = args.loopTest;
        auto imageResult = processImagePath(args.imageFilePath, loopTest);
        imageFiles = imageResult.first;
        loopTest = imageResult.second;

        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(args.modelPath, io);
        model_path_ = args.modelPath;

        if (!dxapp::minversionforRTandCompiler(&ie)) {
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
        auto postprocessor_uptr = factory_->createPostprocessor(input_width, input_height);
        auto postprocessor = std::shared_ptr<typename decltype(postprocessor_uptr)::element_type>(std::move(postprocessor_uptr));
        auto visualizer = factory_->createVisualizer();

        std::cout << "[DXAPP] [INFO] Task: " << factory_->getTaskType() << std::endl;
        std::cout << "[DXAPP] [INFO] Model loaded: " << args.modelPath << std::endl;
        std::cout << "[DXAPP] [INFO] Model input size (WxH): " << input_width << "x" << input_height << std::endl;
        std::cout << std::endl;

        size_t input_size = ie.GetInputSize();
        std::vector<std::vector<uint8_t>> input_buffers(ASYNC_BUFFER_SIZE, std::vector<uint8_t>(input_size));

        std::string run_dir;

        std::cout << "[DXAPP] [INFO] Starting async inference..." << std::endl;
        if (args.no_display) std::cout << "Processing... Only FPS will be displayed." << std::endl;

        cv::Mat display_image;

        std::thread displayThr([this, &visualizer, &args]() {
            displayThread(*visualizer, args.no_display);
        });

        int buffer_index = 0;
        ie.RegisterCallback([this, postprocessor](dxrt::TensorPtrs& outputs, void* user_data) -> int {  // NOSONAR(cpp:S5008)
            auto* ud_ptr = static_cast<AsyncUserData*>(user_data);
            if (!ud_ptr) return 0;
            auto t_post_start = std::chrono::high_resolution_clock::now();

            std::vector<EmbeddingResult> results;
            try { results = postprocessor->process(outputs, ud_ptr->ctx); }
            catch (const std::exception& e) { std::cerr << "[DXAPP] [ERROR] Postprocess error: " << e.what() << std::endl; }

            auto t_post_end = std::chrono::high_resolution_clock::now();
            double t_postprocess = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();
            { std::lock_guard<std::mutex> lock(metrics_.metrics_mutex); metrics_.sum_postprocess += t_postprocess; }

            auto now = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                // Measure inference time (submit → callback)
                double t_inf = std::chrono::duration<double, std::milli>(now - ud_ptr->submit_ts).count();
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

            AsyncEmbeddingDisplayArgs display_args;
            display_args.original_frame = std::make_shared<cv::Mat>(ud_ptr->display_frame);
            display_args.results = std::make_shared<std::vector<EmbeddingResult>>(std::move(results));
            display_args.t_postprocess = t_postprocess;
            display_args.ctx = ud_ptr->ctx;
            display_args.save_path = ud_ptr->save_path;
            display_args.frame_index = ud_ptr->frame_index;
            // --- Numerical verification dump (DXAPP_VERIFY=1) ---
            verify::dumpVerifyJson(*display_args.results, model_path_,
                "embedding", display_args.original_frame->rows, display_args.original_frame->cols);

            display_queue_.push(std::move(display_args));
            delete ud_ptr;
            return 0;
        });

        auto s_time = std::chrono::high_resolution_clock::now();
        int last_job_id = -1;

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
                PreprocessContext ctx;
                cv::Mat preprocessed;
                auto t_pre_start = std::chrono::high_resolution_clock::now();
                preprocessor->process(img, preprocessed, ctx);
                dxapp::displayResize(img, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
                auto t_pre_end = std::chrono::high_resolution_clock::now();
                {
                    std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                    metrics_.sum_read += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
                    metrics_.sum_preprocess += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
                }
                auto& buf = input_buffers[buffer_index % ASYNC_BUFFER_SIZE];
                fillModelInputBuffer(ie, buf, preprocessed);
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
                ud->frame_index = static_cast<uint64_t>(buffer_index);
                last_job_id = ie.RunAsync(buf.data(), static_cast<void*>(ud.release()));
                buffer_index++;
                processCount++;
                if (!args.no_display && !pollDisplay()) break;
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
                // Pump events only — do NOT consume frames from rendered_queue_ here
                cv::waitKey(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            waitThread.join();
        }
        // Wait for displayThread to finish processing all queued items
        while (!display_queue_.empty() && running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // Give render thread time to push last frame to rendered_queue_
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // For images: show each rendered frame one by one, waiting for user input
        if (!args.no_display) {
            std::vector<cv::Mat> rendered_frames;
            for (int attempts = 0; attempts < 20; ++attempts) {
                cv::Mat f;
                while (rendered_queue_.try_pop(f, std::chrono::milliseconds(100))) {
                    rendered_frames.push_back(std::move(f));
                }
                if (!rendered_frames.empty()) break;
            }
            bool wp_supported = true;
            for (size_t fi = 0; fi < rendered_frames.size() && running_; ++fi) {
                dxapp::showOutput(rendered_frames[fi]);
                if (fi == 0) {
                    cv::waitKey(1);
                    try {
                        double probe = cv::getWindowProperty("Output", cv::WND_PROP_VISIBLE);
                        if (probe < 0.0) wp_supported = false;
                    } catch (const cv::Exception&)  { wp_supported = false; }
                }
                bool is_last = (fi + 1 == rendered_frames.size());
                while (running_) {
                    int k = cv::waitKey(10);
                    if (k == 'q' || k == 27) {
                        if (is_last) { running_ = false; break; }
                        break;  // advance to next image
                    }
                    if (k >= 0) break;
                    if (wp_supported) {
                        try {
                            double v = cv::getWindowProperty("Output", cv::WND_PROP_VISIBLE);
                            if (v < 0.0) { running_ = false; break; }
                        } catch (const cv::Exception&)  { running_ = false; break; }
                    }
                }
            }
        }
        // Frames can still be in flight when the reader hits EOF: a completion
        // callback may not have queued its frame yet, and the display thread may
        // hold buffered ones. Wait until every submitted frame has been rendered
        // (and therefore written) before stopping the consumer — otherwise the
        // tail of a --save video is silently lost. The live-display path never
        // showed this because frames are rendered as they arrive. Bail out if no
        // progress is made for 5 s, so a dropped frame cannot hang shutdown.
        {
            auto last_progress = std::chrono::steady_clock::now();
            int last_rendered = -1;
            while (running_ && !g_interrupted()) {
                int rendered;
                {
                    std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                    rendered = metrics_.render_completed;
                }
                const bool pending = !display_queue_.empty() ||
                                     (args.saveMode && rendered < processCount);
                if (!pending) break;
                if (rendered != last_rendered) {
                    last_rendered = rendered;
                    last_progress = std::chrono::steady_clock::now();
                } else if (std::chrono::steady_clock::now() - last_progress >
                           std::chrono::seconds(5)) {
                    std::cerr << "[DXAPP] [WARN] Output frames stopped draining; "
                                 "saved video may be truncated." << std::endl;
                    break;
                }
                if (!args.no_display) pollDisplay();
                else std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
            std::cout << "\n[DXAPP] [INFO] Interrupted by user (Ctrl+C)" << std::endl;
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
    bool is_image_mode_ = false;
    SafeQueue<AsyncEmbeddingDisplayArgs> display_queue_;
    SafeQueue<cv::Mat> rendered_queue_;  // Rendered frames for main-thread display
    AsyncProfilingMetrics metrics_;

    CommandLineArgs parseCommandLine(int argc, char* argv[]) {
        CommandLineArgs args;
        std::string app_name = factory_->getModelName() + " Embedding Async Example";
        cxxopts::Options options(app_name, app_name + " application usage ");
        options.add_options()
            ("m, model_path", "model file (.dxnn, required)", cxxopts::value<std::string>(args.modelPath))
            ("i, image_path", "input image file path or directory (image only)", cxxopts::value<std::string>(args.imageFilePath))
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
        // Model resolved/validated in Run() via dxapp::resolveAndValidateModel().

        if (args.imageFilePath.empty()) { dxapp::fatal_error("[DXAPP] [ERROR] Please specify an image input source with -i or --image_path."); }
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
            if (imageFiles.empty()) { dxapp::fatal_error("[DXAPP] [ERROR] No image files found."); }
            if (loopTest == -1) loopTest = static_cast<int>(imageFiles.size());
        } else if (fs::is_regular_file(imageFilePath)) {
            imageFiles.push_back(imageFilePath);
            if (loopTest == -1) loopTest = 1;
        } else { dxapp::fatal_error("[DXAPP] [ERROR] Input file not found: " + imageFilePath); }
        return {imageFiles, loopTest};
    }

    void displayThread(IVisualizer<EmbeddingResult>& visualizer, bool no_display) {
        // Render + save + hand one frame to the main-thread display. Extracted so
        // the reorder buffer below can emit frames strictly in submit order.
        auto renderArgs = [&](AsyncEmbeddingDisplayArgs& args) {
            if (!args.original_frame || args.original_frame->empty()) return;
            auto t_render_start = std::chrono::high_resolution_clock::now();
            cv::Mat result_frame = visualizer.draw(*args.original_frame, *args.results, args.ctx);
            auto t_render_end = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                metrics_.sum_render += std::chrono::duration<double, std::milli>(t_render_end - t_render_start).count();
                metrics_.render_completed++;
            }
            if (!result_frame.empty()) {
                dxapp::saveDebugImage(result_frame);
            }
            if (!args.save_path.empty() && !result_frame.empty()) {
                auto t_save_start = std::chrono::high_resolution_clock::now();
                cv::imwrite(args.save_path, result_frame);
                if (verbose_) {
                    std::cout << "\n[DXAPP] [INFO] Saved output image: " << fs::absolute(args.save_path).string() << std::endl;
                }
                auto t_save_end = std::chrono::high_resolution_clock::now();
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                metrics_.sum_save += std::chrono::duration<double, std::milli>(t_save_end - t_save_start).count();
            }
            // Push rendered frame for main-thread display (imshow must run on main thread for Qt)
            if (!no_display && !result_frame.empty()) {
                rendered_queue_.push(result_frame.clone());
            }
        };

        // In-order display: completion order is not submission order, and frames
        // are written to the VideoWriter as they are rendered. See frame_reorder.hpp.
        FrameReorderBuffer<AsyncEmbeddingDisplayArgs> reorder(metrics_.max_inflight * 2);

        while (running_ || !display_queue_.empty()) {
            AsyncEmbeddingDisplayArgs args;
            if (!display_queue_.try_pop(args, std::chrono::milliseconds(100))) continue;
            reorder.push(std::move(args), renderArgs);
        }
        reorder.drain(renderArgs);
    }

    /** Poll rendered_queue_ and display on main thread. Returns false if user requested quit. */
    bool pollDisplay() {
        // In image mode, do NOT consume rendered frames here.
        // All image display is handled by the post-loop wait section so that
        // async results (which may arrive out of order) are shown sequentially.
        if (is_image_mode_) {
            cv::waitKey(1);
            return true;
        }
        cv::Mat frame;
        if (rendered_queue_.try_pop(frame, std::chrono::milliseconds(1))) {
            auto display_start = std::chrono::high_resolution_clock::now();
            dxapp::showOutput(frame);
            auto display_end = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                metrics_.sum_display += std::chrono::duration<double, std::milli>(display_end - display_start).count();
                metrics_.display_completed++;
            }
            if (!window_shown_) {
                window_shown_ = true;
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
        if (metrics_.render_completed > 0 && metrics_.sum_render > 0) {
            double avg_render = metrics_.sum_render / metrics_.render_completed;
            printRow("Render", avg_render, avg_render > 0 ? 1000.0/avg_render : 0.0);
        }
        if (save_on && metrics_.sum_save > 0) {
            double avg_save = metrics_.sum_save / metrics_.infer_completed;
            printRow("Save", avg_save, avg_save > 0 ? 1000.0/avg_save : 0.0);
        }
        if (metrics_.display_completed > 0 && metrics_.sum_display > 0) {
            double avg_display = metrics_.sum_display / metrics_.display_completed;
            printRow("Display", avg_display, avg_display > 0 ? 1000.0/avg_display : 0.0);
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

#endif  // ASYNC_EMBEDDING_RUNNER_HPP
