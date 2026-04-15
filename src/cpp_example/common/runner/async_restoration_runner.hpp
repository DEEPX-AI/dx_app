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
#include "common/utility/common_util.hpp"
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
    AsyncRestorationDisplayArgs() = default;
    AsyncRestorationDisplayArgs(const AsyncRestorationDisplayArgs&) = default;
    AsyncRestorationDisplayArgs& operator=(const AsyncRestorationDisplayArgs&) = default;
    AsyncRestorationDisplayArgs(AsyncRestorationDisplayArgs&&) noexcept = default;
    AsyncRestorationDisplayArgs& operator=(AsyncRestorationDisplayArgs&&) noexcept = default;
};

template <typename FactoryT>
class AsyncRestorationRunner {
    bool verbose_ = false;

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
            std::cout << "[INFO] --verbose: This task produces image-based output. "
                         "Use --save or display mode to view results." << std::endl;
        }
        bool save_output = args.saveMode;
        std::string save_dir = args.saveDir;

        // Apply default sample image if no input specified
        if (args.imageFilePath.empty() && args.videoFile.empty() && args.cameraIndex < 0 && args.rtspUrl.empty()) {
            args.imageFilePath = dxapp::getDefaultSampleImage(factory_->getTaskType());
            std::cout << "[INFO] No input specified. Using default sample: " << args.imageFilePath << std::endl;
        }
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
            factory_->loadConfig(config);
        }

        auto preprocessor = factory_->createPreprocessor(input_width, input_height);
        // Use shared_ptr so the callback lambda can capture by value and extend lifetime
        // past ie.Wait() which may return before the background callback thread completes.
        auto postprocessor_uptr = factory_->createPostprocessor(input_width, input_height);
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
            std::string source_info;
            if (args.cameraIndex >= 0) source_info = "Camera index: " + std::to_string(args.cameraIndex);
            else if (!args.rtspUrl.empty()) source_info = "RTSP URL: " + args.rtspUrl;
            else { source_info = "Video file: " + args.videoFile; }
            if (args.verbose) {
                std::cout << "[INFO] " << source_info << std::endl;
                std::cout << "[INFO] Input source resolution (WxH): " << frame_width << "x" << frame_height << std::endl;
                std::cout << "[INFO] Input source FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
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
            catch (const std::exception& e) { std::cerr << "[DXAPP] [ER] Postprocess error: " << e.what() << std::endl; }

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
                    processFrameAsync(frame, frame_params);
                    if (!args.no_display && !pollDisplay()) break;
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
    SafeQueue<AsyncRestorationDisplayArgs> display_queue_;
    SafeQueue<cv::Mat> rendered_queue_;  // Rendered frames for main-thread display
    AsyncProfilingMetrics metrics_;

    // SR model detection (set by probeModel)
    bool is_sr_ = false;
    int tile_w_ = 0, tile_h_ = 0;
    int scale_x_ = 1, scale_y_ = 1;
    int out_tile_w_ = 0, out_tile_h_ = 0;
    int sr_lr_w_ = 0, sr_lr_h_ = 0;

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
        p.last_job_id = p.ie.RunAsync(buf.data(), user_data);
        p.buffer_index++;
        p.processCount++;
    }

    /** Super-Resolution: synchronous tiled inference path. */
    void processFrameSR(const cv::Mat& frame, dxrt::InferenceEngine& ie, int& processCount,
                         const std::string& save_path = "") {
        auto t_pre_start = std::chrono::high_resolution_clock::now();
        int lr_h = static_cast<int>(std::round(
            static_cast<double>(sr_lr_w_) * frame.rows / frame.cols));
        lr_h = ((lr_h + tile_h_ - 1) / tile_h_) * tile_h_;
        if (lr_h <= 0) lr_h = tile_h_ * 10;

        cv::Mat lr_bgr;
        cv::resize(frame, lr_bgr, cv::Size(sr_lr_w_, lr_h));
        cv::Mat lr_gray;
        cv::cvtColor(lr_bgr, lr_gray, cv::COLOR_BGR2GRAY);
        auto t_pre_end = std::chrono::high_resolution_clock::now();

        int out_w = sr_lr_w_ * scale_x_;
        int out_h = lr_h * scale_y_;
        cv::Mat sr_y(out_h, out_w, CV_8UC1, cv::Scalar(0));
        int tiles_x = sr_lr_w_ / tile_w_;
        int tiles_y = lr_h / tile_h_;
        int tiles_done = 0;

        auto ti0 = std::chrono::high_resolution_clock::now();
        for (int ty = 0; ty < tiles_y; ++ty) {
            for (int tx = 0; tx < tiles_x; ++tx) {
                cv::Mat tile = lr_gray(cv::Rect(tx*tile_w_, ty*tile_h_, tile_w_, tile_h_)).clone();
                auto tile_out = ie.Run(tile.data, nullptr, nullptr);
                if (tile_out.empty()) continue;
                const float* data = static_cast<const float*>(tile_out[0]->data());
                if (!data) continue;
                copyTilePixels(data, tx * out_tile_w_, ty * out_tile_h_, sr_y);
                ++tiles_done;
            }
        }
        auto ti1 = std::chrono::high_resolution_clock::now();

        auto t_post_start = std::chrono::high_resolution_clock::now();
        cv::Mat canvas = buildSRCanvas(lr_bgr, sr_y, out_w, out_h, lr_h, tiles_done);
        auto t_post_end = std::chrono::high_resolution_clock::now();

        AsyncRestorationDisplayArgs dargs;
        dargs.prerendered_frame = canvas;
        dargs.save_path = save_path;
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
    cv::Mat buildSRCanvas(const cv::Mat& lr_bgr, const cv::Mat& sr_y,
                          int out_w, int out_h, int lr_h, int tiles_done) const {
        cv::Mat lr_ycrcb;
        cv::cvtColor(lr_bgr, lr_ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> ch;
        cv::split(lr_ycrcb, ch);
        cv::Mat cr_up, cb_up;
        cv::resize(ch[1], cr_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::resize(ch[2], cb_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat ycrcb_merged;
        cv::merge(std::vector<cv::Mat>{sr_y, cr_up, cb_up}, ycrcb_merged);
        cv::Mat sr_bgr;
        cv::cvtColor(ycrcb_merged, sr_bgr, cv::COLOR_YCrCb2BGR);

        cv::Mat lr_upscaled;
        cv::resize(lr_bgr, lr_upscaled, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat canvas(out_h, out_w*2+4, CV_8UC3, cv::Scalar(0,0,0));
        lr_upscaled.copyTo(canvas(cv::Rect(0, 0, out_w, out_h)));
        sr_bgr.copyTo(canvas(cv::Rect(out_w+4, 0, out_w, out_h)));
        cv::putText(canvas, cv::format("Bicubic (%dx%d)", sr_lr_w_, lr_h),
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
                std::cout << "[INFO] Multi-channel model (C=" << input_channels
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
            const int TARGET_TILES_W = 20;
            sr_lr_w_ = tile_w_ * TARGET_TILES_W;
            // height will be determined per-frame from aspect ratio;
            // store a reasonable default here (updated per-frame if needed)
            sr_lr_h_ = tile_h_ * 12;  // default ~240px, overridden below
            // Mark: actual lr_h is computed per-frame in processFrameAsync
            if (verbose_) {
                std::cout << "[INFO] SR model detected (scale x" << scale_x_
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
            ("h, help", "print usage");
        auto cmd = options.parse(argc, argv);
        if (cmd.count("help")) { std::cout << options.help() << std::endl; exit(0); }
        return args;
    }

    void validateArguments(const CommandLineArgs& args) {
        if (args.modelPath.empty()) { dxapp::fatal_error("[ERROR] Model path is required. Use -m or --model_path option.\n"
                "        -> Download:  ./setup.sh --models <model_name>\n"
                "        -> Or use:    ./run_demo.sh  (auto-downloads demo models)"); }
        // Auto-download model if not found
        if (!dxapp::fileExists(args.modelPath)) {
            if (!dxapp::autoDownloadModel(args.modelPath)) {
                std::string stem = fs::path(args.modelPath).stem().string();
                dxapp::fatal_error("[ERROR] Model file not found: " + args.modelPath + "\n"
                    "        -> Download:  ./setup.sh --models " + stem + "\n"
                    "        -> Or use:    ./run_demo.sh  (auto-downloads demo models)");
            }
            std::cout << "[INFO] Model downloaded successfully: " << args.modelPath << std::endl;
        }

        int sourceCount = 0;
        if (!args.imageFilePath.empty()) sourceCount++;
        if (!args.videoFile.empty()) sourceCount++;
        if (args.cameraIndex >= 0) sourceCount++;
        if (!args.rtspUrl.empty()) sourceCount++;
        if (sourceCount != 1) { dxapp::fatal_error("[ERROR] Please specify exactly one input source."); }
        // Auto-download video if not found
        if (!args.videoFile.empty() && !dxapp::fileExists(args.videoFile)) {
            if (!dxapp::autoDownloadVideos() || !dxapp::fileExists(args.videoFile)) {
                dxapp::fatal_error("[ERROR] Video file not found: " + args.videoFile + "\n"
                    "        -> Download videos: ./setup_sample_videos.sh");
            }
            std::cout << "[INFO] Video downloaded successfully: " << args.videoFile << std::endl;
        }

        // Validate that --video is not given an image file
        if (!args.videoFile.empty()) {
            std::string ext = fs::path(args.videoFile).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff") {
                dxapp::fatal_error("[ERROR] Image file detected for --video (-v) option. "
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

    /** Write a result frame to the video writer, resizing if dimensions differ. */
    static void writeVideoFrame(cv::VideoWriter& writer, const cv::Mat& frame) {
        if (frame.cols != static_cast<int>(SHOW_WINDOW_SIZE_W) ||
            frame.rows != static_cast<int>(SHOW_WINDOW_SIZE_H)) {
            cv::Mat write_frame;
            cv::resize(frame, write_frame, cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
            dxapp::writeToVideo(writer, write_frame);
        } else {
            dxapp::writeToVideo(writer, frame);
        }
    }

    void displayThread(IVisualizer<RestorationResult>& visualizer, bool no_display,
                       bool save_on, cv::VideoWriter& writer) {
        while (running_) {
            AsyncRestorationDisplayArgs args;
            if (!display_queue_.try_pop(args, std::chrono::milliseconds(100))) continue;
            cv::Mat result_frame;
            auto t_render_start = std::chrono::high_resolution_clock::now();
            if (!args.prerendered_frame.empty()) {
                result_frame = args.prerendered_frame;
            } else {
                if (!args.original_frame || args.original_frame->empty()) continue;
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
                    std::cout << "\n[INFO] Saved output image: " << fs::absolute(args.save_path).string() << std::endl;
                }
            }
            if (save_on && !result_frame.empty()) writeVideoFrame(writer, result_frame);
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

#endif  // ASYNC_RESTORATION_RUNNER_HPP
