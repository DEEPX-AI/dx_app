/**
 * @file async_restoration_runner.hpp
 * @brief Asynchronous image restoration runner using factory pattern
 *
 * Provides a generic async runner that accepts any IRestorationFactory implementation.
 */

#ifndef ASYNC_RESTORATION_RUNNER_HPP
#define ASYNC_RESTORATION_RUNNER_HPP

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
#include "common/utility/colorspace.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/sr_tiling.hpp"
#include "common/utility/frame_reorder.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/verify_serialize.hpp"
#include "async_detection_runner.hpp"

namespace dxapp {

struct AsyncRestorationDisplayArgs {
    std::shared_ptr<std::vector<RestorationResult>> results;
    std::shared_ptr<cv::Mat> original_frame;
    cv::Mat prerendered_frame;   // if non-empty, used directly (SR tiled path)
    std::string save_path;
    double t_read = 0.0;
    double t_preprocess = 0.0;
    double t_inference = 0.0;
    double t_postprocess = 0.0;
    PreprocessContext ctx;
    uint64_t frame_index = 0;  // Monotonic submit order, for in-order display
    AsyncRestorationDisplayArgs() = default;
    AsyncRestorationDisplayArgs(const AsyncRestorationDisplayArgs&) = default;
    AsyncRestorationDisplayArgs& operator=(const AsyncRestorationDisplayArgs&) = default;
    AsyncRestorationDisplayArgs(AsyncRestorationDisplayArgs&&) noexcept = default;
    AsyncRestorationDisplayArgs& operator=(AsyncRestorationDisplayArgs&&) noexcept = default;
};

template <typename FactoryT>
class AsyncRestorationRunner {
    bool verbose_ = false;
    /// Tile halo sources; -1 = not given. See srtiling::resolveHaloFrom.
    int cli_halo_ = -1;   ///< --sr-tile-halo (bound directly by parseCommandLine)
    int cfg_halo_ = -1;   ///< config.json "sr_tile_halo"
    int sr_halo_ = 0;     ///< resolved halo (0 is a valid value, hence the flag)
    bool sr_halo_resolved_ = false;
    bool tiling_logged_ = false;  ///< tiles-per-frame line is printed once per run

public:
    explicit AsyncRestorationRunner(std::unique_ptr<FactoryT> factory)
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
        bool save_output = args.saveMode;
        std::string save_dir = args.saveDir;

        // Apply default sample image if no input specified
        if (args.imageFilePath.empty() && args.videoFile.empty() && args.cameraIndex < 0 && args.rtspUrl.empty()) {
            args.imageFilePath = dxapp::getDefaultSampleImage(factory_->getTaskType(),
                                                             factory_->getModelName());
            std::cout << "[DXAPP] [INFO] No input specified. Using default sample: " << args.imageFilePath << std::endl;
        }
        dxapp::resolveAndValidateModel(args.modelPath, argv[0]);
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
            return -1;
        }

        auto input_shape = ie.GetInputs().front().shape();
        int input_height, input_width;
        parseInputShape(input_shape, input_width, input_height);

        // Detect layout before probe so channel count is read correctly
        bool is_nhwc = isInputNHWC(input_shape);

        // Probe model to detect SR vs denoising
        probeModel(ie, input_width, input_height, is_nhwc);

        // Detect if model expects float input (e.g. Zero-DCE)
        bool is_float_input = (ie.GetInputs().front().type() == dxrt::DataType::FLOAT);
        int input_channels = 1;
        if (input_shape.size() >= 4) {
            input_channels = static_cast<int>(is_nhwc ? input_shape[3] : input_shape[1]);
        }

        // Load model configuration if provided
        if (!args.configPath.empty()) {
            dxapp::ModelConfig config(args.configPath);
            cfg_halo_ = config.get<int>("sr_tile_halo", -1);
            factory_->loadConfig(config);
        }

        // Resolve the tile halo here, on the main thread: the tiled path runs in a
        // worker, and rejecting a bad value from there would abort the process
        // instead of printing one actionable line. Non-SR restoration models have
        // no tiles, so they skip it (and ignore --sr-tile-halo).
        if (is_sr_) {
            resolveTileHalo(tile_h_, tile_w_);
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

        cv::VideoCapture video;
        cv::VideoWriter writer;
        std::string run_dir;
        std::string video_save_path;

        if (!is_image) {
            if (!openVideoCapture(video, args)) {
                std::cerr << "[DXAPP] [ERROR] Failed to open input source." << std::endl;
                return -1;
            }
            auto frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
            auto frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps = video.get(cv::CAP_PROP_FPS);
            std::string source_info;
            if (args.cameraIndex >= 0) source_info = "Camera index: " + std::to_string(args.cameraIndex);
            else if (!args.rtspUrl.empty()) source_info = "RTSP URL: " + args.rtspUrl;
            else { source_info = "Video file: " + args.videoFile; }
            if (args.verbose) {
                std::cout << "[DXAPP] [INFO] " << source_info << std::endl;
                std::cout << "[DXAPP] [INFO] Input source resolution (WxH): " << frame_width << "x" << frame_height << std::endl;
                std::cout << "[DXAPP] [INFO] Input source FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
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
                    std::cerr << "[DXAPP] [ERROR] Failed to open video writer." << std::endl;
                    return -1;
                }
            }
        }

        std::cout << "[DXAPP] [INFO] Starting async inference..." << std::endl;
        if (args.no_display) std::cout << "Processing... Only FPS will be displayed." << std::endl;

        cv::Mat display_image;

        std::thread displayThr([this, &visualizer, &args, &writer]() {
            displayThread(*visualizer, args.no_display, args.saveMode, writer);
        });

        int buffer_index = 0;
        ie.RegisterCallback([this, postprocessor](dxrt::TensorPtrs& outputs, void* user_data) -> int {  // NOSONAR(cpp:S5008)
            // user_data is null for tiled SR path (ie.Run called without user_data)
            if (!user_data) return 0;
            auto* ud = static_cast<AsyncUserData*>(user_data);
            auto t_post_start = std::chrono::high_resolution_clock::now();

            std::vector<RestorationResult> results;
            try { results = postprocessor->process(outputs, ud->ctx); }
            catch (const std::exception& e) { std::cerr << "[DXAPP] [ERROR] Postprocess error: " << e.what() << std::endl; }

            auto t_post_end = std::chrono::high_resolution_clock::now();
            double t_postprocess = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();
            { std::lock_guard<std::mutex> lock(metrics_.metrics_mutex); metrics_.sum_postprocess += t_postprocess; }

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

            AsyncRestorationDisplayArgs display_args;
            display_args.original_frame = std::make_shared<cv::Mat>(ud->display_frame);
            display_args.results = std::make_shared<std::vector<RestorationResult>>(std::move(results));
            display_args.t_postprocess = t_postprocess;
            display_args.ctx = ud->ctx;
            display_args.save_path = std::move(ud->save_path);
            display_args.frame_index = ud->frame_index;
            // --- Numerical verification dump (DXAPP_VERIFY=1) ---
            verify::dumpVerifyJson(*display_args.results, model_path_,
                "restoration", display_args.original_frame->rows, display_args.original_frame->cols);

            display_queue_.push(std::move(display_args));
            delete ud;
            return 0;
        });

        auto s_time = std::chrono::high_resolution_clock::now();
        int last_job_id = -1;

        AsyncFrameParams frame_params{display_image, *preprocessor, input_buffers,
            buffer_index, ie, last_job_id, processCount, is_float_input, is_nhwc};

        if (is_image) {
            // Prepare run directory for image-mode saves
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
                {
                    std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                    metrics_.sum_read += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
                }
                const std::string& cur_img_path = imageFiles[i % imageFiles.size()];
                if (is_sr_) {
                    std::string sr_save_path;
                    if (!run_dir.empty()) {
                        sr_save_path = dxapp::buildPerImageSavePath(run_dir, factory_->getModelName() + "_async", cur_img_path, i);
                    }
                    processFrameSR(img, frame_params.ie, frame_params.processCount, sr_save_path);
                } else {
                    std::string save_path;
                    if (!run_dir.empty()) {
                        save_path = dxapp::buildPerImageSavePath(run_dir, factory_->getModelName() + "_async", cur_img_path, i);
                    }
                    submitDenoisingFrame(img, display_image, frame_params, std::move(save_path));
                }
                if (!args.no_display && !pollDisplay()) break;
            }
        } else {
            for (int loop_idx = 0; loop_idx < loopTest && running_ && !g_interrupted(); ++loop_idx) {
                if (loopTest > 1) {
                    if (args.verbose) {
                        std::cout << "\n" << std::string(50, '=') << std::endl;
                        std::cout << "[DXAPP] [INFO] Loop " << (loop_idx + 1) << "/" << loopTest << std::endl;
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
                    processFrameAsync(frame, frame_params);
                    if (!args.no_display && !pollDisplay()) break;
                }
                // Reopen video for next loop
                if (loop_idx + 1 >= loopTest || args.videoFile.empty()) continue;
                video.release();
                video.open(args.videoFile);
                if (!video.isOpened()) {
                    std::cerr << "[DXAPP] [ERROR] Failed to reopen video for loop " << (loop_idx + 2) << std::endl;
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
                if (!args.no_display && !pollDisplay()) break;
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
        if (writer.isOpened()) {
            writer.release();
            if (!video_save_path.empty()) {
                if (args.verbose) {
                    std::cout << "\n[DXAPP] [INFO] Saved output video: " << fs::absolute(video_save_path).string() << std::endl;
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
    SafeQueue<AsyncRestorationDisplayArgs> display_queue_;
    SafeQueue<cv::Mat> rendered_queue_;  // Rendered frames for main-thread display
    AsyncProfilingMetrics metrics_;

    // SR model detection (set by probeModel)
    bool is_sr_ = false;
    int tile_w_ = 0, tile_h_ = 0;
    int scale_x_ = 1, scale_y_ = 1;
    int out_tile_w_ = 0, out_tile_h_ = 0;

    /** Bundled mutable session state for async frame processing. */
    struct AsyncFrameParams {
        cv::Mat& display_image;
        IPreprocessor& preprocessor;
        std::vector<std::vector<uint8_t>>& input_buffers;
        int& buffer_index;
        dxrt::InferenceEngine& ie;
        int& last_job_id;
        int& processCount;
        bool is_float_input;
        bool is_nhwc;
    };

    /** Dispatch a frame to SR tiled or denoising async path. */
    void processFrameAsync(const cv::Mat& frame, AsyncFrameParams& p) {
        if (is_sr_) {
            processFrameSR(frame, p.ie, p.processCount);
        } else {
            processFrameDenoiseAsync(frame, p);
        }
    }

    /** Denoising: preprocess and submit a single frame to async inference. */
    void submitDenoisingFrame(const cv::Mat& img, cv::Mat& display_image,
                              AsyncFrameParams& p, std::string save_path) {
        auto t_pre_start = std::chrono::high_resolution_clock::now();
        PreprocessContext ctx;
        cv::Mat preprocessed;
        p.preprocessor.process(img, preprocessed, ctx);
        dxapp::displayResize(img, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
        auto t_pre_end = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
            metrics_.sum_preprocess += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
        }
        auto& buf = p.input_buffers[p.buffer_index % ASYNC_BUFFER_SIZE];
        if (p.is_float_input && !preprocessed.empty()) {
            auto float_data = convertToFloatBuffer(preprocessed, p.is_nhwc);
            std::memcpy(buf.data(), float_data.data(), float_data.size() * sizeof(float));
        } else {
            std::memcpy(buf.data(), preprocessed.data, preprocessed.total() * preprocessed.elemSize());
        }
        auto ud = std::make_unique<AsyncUserData>(AsyncUserData{display_image.clone(), ctx, std::move(save_path), {}});
        void* user_data = ud.release();
        metrics_.waitForSlot();
        updateInflightMetrics();
        static_cast<AsyncUserData*>(user_data)->submit_ts = std::chrono::high_resolution_clock::now();
        static_cast<AsyncUserData*>(user_data)->frame_index = static_cast<uint64_t>(p.buffer_index);
        p.last_job_id = p.ie.RunAsync(buf.data(), user_data);
        p.buffer_index++;
        p.processCount++;
    }

    /** Super-Resolution: synchronous tiled inference path. */
    void processFrameSR(const cv::Mat& frame, dxrt::InferenceEngine& ie, int& processCount,
                         const std::string& save_path = "") {
        auto t_pre_start = std::chrono::high_resolution_clock::now();
        int orig_w = frame.cols;
        int orig_h = frame.rows;

        // Halo-aware tiling: windows overlap by `halo`, only valid centres are
        // stitched, so with halo >= the receptive-field radius (4 px for ESPCN) no
        // tile seams remain. See common/utility/sr_tiling.hpp.
        // --sr-tile-halo > config.json "sr_tile_halo" > env > default.
        const int halo = resolveTileHalo(tile_h_, tile_w_);
        int padded_h = 0, padded_w = 0;
        std::vector<dxapp::srtiling::TilePlan> plans;
        dxapp::srtiling::planTiles(orig_h, orig_w, tile_h_, tile_w_, halo,
                                   padded_h, padded_w, plans);
        // Report the tiling plan once: at a fixed 17x17 input the per-frame cost is
        // driven by how many tiles the frame is cut into, i.e. how many inferences
        // run per frame. Printed once (the plan only changes with input size/halo),
        // so a stream does not repeat it every frame.
        if (!tiling_logged_) {
            tiling_logged_ = true;
            std::cout << "[DXAPP] [INFO] SR tiling: " << orig_w << "x" << orig_h
                      << " -> " << plans.size() << " tiles of " << tile_w_ << "x"
                      << tile_h_ << " (halo=" << halo << " px), so "
                      << plans.size() << " inferences per frame" << std::endl;
        }
        if (plans.size() > 400) {
            std::cerr << "[DXAPP] [WARN] SR: large input (" << orig_w << "x" << orig_h
                      << ") produces " << plans.size() << " tiles; processing may be slow.\n";
        }

        cv::Mat lr_bgr;
        cv::copyMakeBorder(frame, lr_bgr, 0, padded_h - orig_h, 0, padded_w - orig_w,
                   cv::BORDER_REPLICATE);
        // ESPCN is trained on MATLAB rgb2ycbcr Y, so feed limited-range Y
        // rather than OpenCV's full-range grayscale.
        cv::Mat lr_gray;
        dxapp::colorspace::bgrToYLimited(lr_bgr, lr_gray);
        auto t_pre_end = std::chrono::high_resolution_clock::now();

        int padded_out_w = padded_w * scale_x_;
        int padded_out_h = padded_h * scale_y_;
        int out_w = orig_w * scale_x_;
        int out_h = orig_h * scale_y_;
        cv::Mat sr_y_padded;

        // Blocking Run per tile — NOT the pipelined variant. This runner has a
        // callback registered on `ie` (see RegisterCallback in run()), which
        // consumes async outputs, so RunAsync + Wait() would return empty tensors
        // for every tile. The sync restoration runner has no callback and does use
        // runTilesPipelined().
        auto ti0 = std::chrono::high_resolution_clock::now();
        std::vector<dxrt::TensorPtrs> tile_outputs;
        dxapp::srtiling::runTilesBlocking(
            ie, lr_gray, plans, tile_w_, tile_h_, tile_outputs);
        int tiles_done = dxapp::srtiling::assembleTiles(
            plans, tile_outputs, padded_out_h, padded_out_w,
            scale_y_, scale_x_, out_tile_w_, sr_y_padded);
        auto ti1 = std::chrono::high_resolution_clock::now();

        auto t_post_start = std::chrono::high_resolution_clock::now();
        cv::Mat sr_y = sr_y_padded(cv::Rect(0, 0, out_w, out_h)).clone();
        cv::Mat orig_bgr = lr_bgr(cv::Rect(0, 0, orig_w, orig_h));
        cv::Mat sr_only;
        cv::Mat canvas = buildSRCanvas(orig_bgr, sr_y, out_w, out_h, orig_w, orig_h,
                                       tiles_done, &sr_only);
        // Write <name>_output_only.<ext> — the raw SR output, no Bicubic panel and
        // no drawn labels — for both --save and DXAPP_SAVE_IMAGE, so this variant
        // produces the same file set as the sync runner and the Python examples.
        {
            auto write_output_only = [&](const std::string& p) {
                if (p.empty() || sr_only.empty()) return;
                std::size_t dot = p.find_last_of('.');
                std::string out = (dot == std::string::npos)
                    ? p + "_output_only"
                    : p.substr(0, dot) + "_output_only" + p.substr(dot);
                cv::imwrite(out, sr_only);
            };
            write_output_only(save_path);
            const char* sv = std::getenv("DXAPP_SAVE_IMAGE");
            if (sv && *sv && std::string(sv) != save_path)
                write_output_only(std::string(sv));
        }
        auto t_post_end = std::chrono::high_resolution_clock::now();

        AsyncRestorationDisplayArgs dargs;
        dargs.prerendered_frame = canvas;
        dargs.save_path = save_path;
        // This path renders on the caller's thread and is already in submit order,
        // but the display thread keys its reorder buffer on frame_index, so the
        // index must still be unique and monotonic here.
        dargs.frame_index = static_cast<uint64_t>(processCount);
        display_queue_.push(std::move(dargs));
        {
            std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
            auto now = std::chrono::high_resolution_clock::now();
            if (metrics_.first_inference) {
                metrics_.infer_first_ts = now; metrics_.inflight_last_ts = now; metrics_.first_inference = false;
            }
            metrics_.infer_last_ts = now;
            metrics_.infer_completed++;
            metrics_.sum_preprocess += std::chrono::duration<double,std::milli>(t_pre_end-t_pre_start).count();
            metrics_.sum_inference += std::chrono::duration<double,std::milli>(ti1-ti0).count();
            metrics_.sum_postprocess += std::chrono::duration<double,std::milli>(t_post_end-t_post_start).count();
        }
        processCount++;
    }

    /** Copy a single tile's float output pixels to the SR Y-channel image. */
    void copyTilePixels(const float* data, int dst_x, int dst_y, cv::Mat& sr_y) const {
        for (int py = 0; py < out_tile_h_; ++py)
            for (int px = 0; px < out_tile_w_; ++px) {
                float v = std::max(0.0f, std::min(1.0f, data[py*out_tile_w_+px]));
                sr_y.at<uchar>(dst_y+py, dst_x+px) = static_cast<uchar>(v*255.0f+0.5f);
            }
    }

    /** Build side-by-side bicubic vs SR canvas for display. */
    /** Build the side-by-side canvas. If `sr_bgr_out` is given it receives the
     *  label-free SR image, which is what `<name>_output_only` must contain —
     *  cropping it back out of the canvas would include the drawn label. */
    cv::Mat buildSRCanvas(const cv::Mat& lr_bgr, const cv::Mat& sr_y,
                          int out_w, int out_h, int lr_w, int lr_h, int tiles_done,
                          cv::Mat* sr_bgr_out = nullptr) const {
        // sr_y is limited-range — keep the chroma planes on the same convention.
        cv::Mat lr_ycrcb;
        dxapp::colorspace::bgrToYCrCbLimited(lr_bgr, lr_ycrcb);
        std::vector<cv::Mat> ch;
        cv::split(lr_ycrcb, ch);
        cv::Mat cr_up, cb_up;
        cv::resize(ch[1], cr_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::resize(ch[2], cb_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat ycrcb_merged;
        cv::merge(std::vector<cv::Mat>{sr_y, cr_up, cb_up}, ycrcb_merged);
        cv::Mat sr_bgr;
        dxapp::colorspace::ycrcbLimitedToBgr(ycrcb_merged, sr_bgr);
        if (sr_bgr_out != nullptr) *sr_bgr_out = sr_bgr.clone();

        cv::Mat lr_upscaled;
        cv::resize(lr_bgr, lr_upscaled, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat canvas(out_h, out_w*2+4, CV_8UC3, cv::Scalar(0,0,0));
        lr_upscaled.copyTo(canvas(cv::Rect(0, 0, out_w, out_h)));
        sr_bgr.copyTo(canvas(cv::Rect(out_w+4, 0, out_w, out_h)));
        cv::putText(canvas, cv::format("Bicubic (%dx%d)", lr_w, lr_h),
                    cv::Point(10,25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,200,255), 2);
        cv::putText(canvas,
                    cv::format("ESPCN x%d (%dx%d, %d tiles)", scale_x_, out_w, out_h, tiles_done),
                    cv::Point(out_w+14,25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,100), 2);
        return canvas;
    }

    /** Denoising: submit frame for async inference. */
    void processFrameDenoiseAsync(const cv::Mat& frame, AsyncFrameParams& p) {
        auto t_pre_start = std::chrono::high_resolution_clock::now();
        PreprocessContext ctx;
        cv::Mat preprocessed;
        p.preprocessor.process(frame, preprocessed, ctx);
        dxapp::displayResize(frame, p.display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
        auto t_pre_end = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
            metrics_.sum_preprocess += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
        }
        auto& buf = p.input_buffers[p.buffer_index % ASYNC_BUFFER_SIZE];

        if (p.is_float_input && !preprocessed.empty()) {
            auto float_data = convertToFloatBuffer(preprocessed, p.is_nhwc);
            std::memcpy(buf.data(), float_data.data(), float_data.size() * sizeof(float));
        } else {
            std::memcpy(buf.data(), preprocessed.data, preprocessed.total() * preprocessed.elemSize());
        }
        auto user_data_ptr = std::make_unique<AsyncUserData>(AsyncUserData{p.display_image.clone(), ctx, std::string(), {}});
        void* user_data = user_data_ptr.release();
        metrics_.waitForSlot();
        updateInflightMetrics();
        static_cast<AsyncUserData*>(user_data)->submit_ts = std::chrono::high_resolution_clock::now();
        static_cast<AsyncUserData*>(user_data)->frame_index = static_cast<uint64_t>(p.buffer_index);
        p.last_job_id = p.ie.RunAsync(buf.data(), user_data);
        p.buffer_index++;
        p.processCount++;
    }

    /** Update in-flight tracking metrics (single producer). */
    void updateInflightMetrics() {
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

    /** Tile halo for the tiled SR path, resolved once per run.
     *
     *  --sr-tile-halo > config.json "sr_tile_halo" > DXAPP_SR_TILE_HALO > default.
     *  A halo the model cannot use is a configuration mistake, so it ends the run
     *  with one actionable line rather than throwing from inside planTiles(). */
    int resolveTileHalo(int tile_h, int tile_w) {
        if (sr_halo_resolved_) return sr_halo_;
        std::string source, error;
        const int halo = dxapp::srtiling::resolveHaloFrom(
            cli_halo_, cfg_halo_, tile_h, tile_w, source, error);
        if (!error.empty()) {
            dxapp::fatal_error("[DXAPP] [ERROR] " + error);
        }
        if (verbose_) {
            std::cout << "[DXAPP] [INFO] SR tile halo: " << halo
                      << " px (from " << source << ")" << std::endl;
        }
        sr_halo_ = halo;
        sr_halo_resolved_ = true;
        return sr_halo_;
    }

    void probeModel(dxrt::InferenceEngine& ie, int input_width, int input_height, bool is_nhwc = false) {
        tile_w_ = input_width;
        tile_h_ = input_height;

        // Check input channels — skip probe for multi-channel (e.g. RGB) models
        auto input_shape = ie.GetInputs().front().shape();
        int input_channels = 1;
        if (input_shape.size() >= 4) {
            input_channels = static_cast<int>(is_nhwc ? input_shape[3] : input_shape[1]);
        }
        if (input_channels > 1) {
            is_sr_ = false;
            if (verbose_) {
                std::cout << "[DXAPP] [INFO] Multi-channel model (C=" << input_channels
                      << "), skipping SR probe." << std::endl;
            }
            return;
        }

        // Run a zero-filled probe tile (single-channel only)
        std::vector<uint8_t> probe_buf(input_width * input_height, 0);
        auto probe_out = ie.Run(probe_buf.data(), nullptr, nullptr);
        int out_h = input_height, out_w = input_width;
        if (!probe_out.empty()) {
            auto shape = probe_out[0]->shape();
            if (shape.size() == 4) { out_h = static_cast<int>(shape[2]); out_w = static_cast<int>(shape[3]); }
            else if (shape.size() == 3) { out_h = static_cast<int>(shape[1]); out_w = static_cast<int>(shape[2]); }
            else if (shape.size() == 2) { out_h = static_cast<int>(shape[0]); out_w = static_cast<int>(shape[1]); }
        }
        scale_x_ = std::max(1, out_w / input_width);
        scale_y_ = std::max(1, out_h / input_height);
        out_tile_w_ = out_w;
        out_tile_h_ = out_h;
        is_sr_ = (scale_x_ > 1 || scale_y_ > 1);
        if (is_sr_) {
            // lr_w / lr_h are now computed per-frame from original dimensions
            if (verbose_) {
                std::cout << "[DXAPP] [INFO] SR model detected (scale x" << scale_x_
                      << "), tiled inference will be used." << std::endl;
            }
        }
    }

    CommandLineArgs parseCommandLine(int argc, char* argv[]) {
        CommandLineArgs args;
        std::string app_name = factory_->getModelName() + " Image Restoration Async Example";
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
            // Bound straight to this runner's own member: the halo is specific to
            // tiled super-resolution, so it stays out of the CommandLineArgs
            // struct that every runner (detection included) shares.
            ("sr-tile-halo", "Tile overlap in LR pixels for tiled super-resolution, "
                             "0..4 (default: 4 = the ESPCN receptive-field radius, the "
                             "most context an output pixel can use; 0 = no overlap, "
                             "fastest but seams appear). Overrides config.json "
                             "'sr_tile_halo' and DXAPP_SR_TILE_HALO.",
             cxxopts::value<int>(cli_halo_)->default_value("-1"))
            ("h, help", "print usage");
        auto cmd = options.parse(argc, argv);
        if (cmd.count("help")) { std::cout << options.help() << std::endl; exit(0); }
        return args;
    }

    void validateArguments(const CommandLineArgs& args) {
        // Model resolved/validated in Run() via dxapp::resolveAndValidateModel().

        int sourceCount = 0;
        if (!args.imageFilePath.empty()) sourceCount++;
        if (!args.videoFile.empty()) sourceCount++;
        if (args.cameraIndex >= 0) sourceCount++;
        if (!args.rtspUrl.empty()) sourceCount++;
        if (sourceCount != 1) { dxapp::fatal_error("[DXAPP] [ERROR] Please specify exactly one input source."); }
        // Explicit input must exist (SDKREQ-529): wrong -v/-i errors out; no auto-download.
        dxapp::requireInputExists(args.videoFile);
        dxapp::requireInputExists(args.imageFilePath);

        // Validate that --video is not given an image file
        if (!args.videoFile.empty()) {
            std::string ext = fs::path(args.videoFile).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff") {
                dxapp::fatal_error("[DXAPP] [ERROR] Image file detected for --video (-v) option. "
                                  "Use --image (-i) for image files.\nUse -h or --help for usage information.");
            }
        }
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

    bool openVideoCapture(cv::VideoCapture& video, const CommandLineArgs& args) {
        if (args.cameraIndex >= 0) video.open(args.cameraIndex);
        else if (!args.rtspUrl.empty()) video.open(args.rtspUrl);
        else video.open(args.videoFile);
        return video.isOpened();
    }

    /** Write a result frame to the video writer, resizing if dimensions differ. */
    static void writeVideoFrame(cv::VideoWriter& writer, const cv::Mat& frame) {
        if (frame.cols != static_cast<int>(SHOW_WINDOW_SIZE_W) ||
            frame.rows != static_cast<int>(SHOW_WINDOW_SIZE_H)) {
            cv::Mat write_frame;
            cv::resize(frame, write_frame, cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            dxapp::writeToVideo(writer, write_frame, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
        } else {
            dxapp::writeToVideo(writer, frame, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
        }
    }

    void displayThread(IVisualizer<RestorationResult>& visualizer, bool no_display,
                       bool save_on, cv::VideoWriter& writer) {
        // Render + save + hand one frame to the main-thread display. Extracted so
        // the reorder buffer below can emit frames strictly in submit order.
        auto renderArgs = [&](AsyncRestorationDisplayArgs& args) {
            cv::Mat result_frame;
            auto t_render_start = std::chrono::high_resolution_clock::now();
            if (!args.prerendered_frame.empty()) {
                result_frame = args.prerendered_frame;
            } else {
                if (!args.original_frame || args.original_frame->empty()) return;
                result_frame = visualizer.draw(*args.original_frame, *args.results, args.ctx);
            }
            auto t_render_end = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(metrics_.metrics_mutex);
                metrics_.sum_render += std::chrono::duration<double, std::milli>(t_render_end - t_render_start).count();
                metrics_.render_completed++;
            }
            if (!args.save_path.empty() && !result_frame.empty()) {
                cv::imwrite(args.save_path, result_frame);
                if (verbose_) {
                    std::cout << "\n[DXAPP] [INFO] Saved output image: " << fs::absolute(args.save_path).string() << std::endl;
                }
            }
            if (save_on && !result_frame.empty()) writeVideoFrame(writer, result_frame);
            dxapp::saveDebugImage(result_frame);
            // Super-resolution standard path (full-color models like RealESRGAN
            // take this path, not the tiled ESPCN path): also save the upscaled
            // output on its own as <name>_output_only.<ext>, next to the
            // side-by-side canvas. The tiled path already does this itself, and
            // sets prerendered_frame with no results, so it won't double-write.
            if (args.results && !args.results->empty() &&
                !(*args.results)[0].restored_image.empty() &&
                factory_->getTaskType() == "super_resolution") {
                const cv::Mat& sr_only = (*args.results)[0].restored_image;
                auto write_output_only = [&sr_only](const std::string& p) {
                    if (p.empty()) return;
                    std::size_t dot = p.find_last_of('.');
                    std::string out = (dot == std::string::npos)
                        ? p + "_output_only"
                        : p.substr(0, dot) + "_output_only" + p.substr(dot);
                    cv::imwrite(out, sr_only);
                };
                write_output_only(args.save_path);
                const char* sv = std::getenv("DXAPP_SAVE_IMAGE");
                if (sv && *sv && std::string(sv) != args.save_path)
                    write_output_only(std::string(sv));
            }
            // Push rendered frame for main-thread display (imshow must run on main thread for Qt)
            if (!no_display && !result_frame.empty()) {
                rendered_queue_.push(result_frame.clone());
            }
        };

        // In-order display: completion order is not submission order, and frames
        // are written to the VideoWriter as they are rendered. See frame_reorder.hpp.
        FrameReorderBuffer<AsyncRestorationDisplayArgs> reorder(metrics_.max_inflight * 2);

        while (running_ || !display_queue_.empty()) {
            AsyncRestorationDisplayArgs args;
            if (!display_queue_.try_pop(args, std::chrono::milliseconds(100))) continue;
            reorder.push(std::move(args), renderArgs);
        }
        reorder.drain(renderArgs);
    }

    /** Poll rendered_queue_ and display on main thread. Returns false if user requested quit. */
    bool pollDisplay() {
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
                // Probe backend: some backends (e.g. GTK2) always return -1
                // for WND_PROP_VISIBLE. Detect this on the first frame so we
                // never falsely interpret -1 as "window closed by user".
                cv::waitKey(1);
                double probe = cv::getWindowProperty("Output", cv::WND_PROP_VISIBLE);
                if (probe < -0.5) {
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
            if (vis <= 0.0) {
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

#endif  // ASYNC_RESTORATION_RUNNER_HPP
