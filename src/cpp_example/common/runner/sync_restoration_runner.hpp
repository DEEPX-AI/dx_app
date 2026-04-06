/**
 * @file restoration_runner.hpp
 * @brief Synchronous image restoration runner using factory pattern
 *
 * Provides a generic runner that accepts any IRestorationFactory implementation.
 */

#ifndef RESTORATION_RUNNER_HPP
#define RESTORATION_RUNNER_HPP

#include <dxrt/dxrt_api.h>
#include <chrono>
#include <cxxopts.hpp>
#include <thread>
#include <experimental/filesystem>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "common/base/i_factory.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/verify_serialize.hpp"
#include "sync_detection_runner.hpp"

namespace dxapp {

template <typename FactoryT>
class SyncRestorationRunner {
    bool verbose_ = false;

public:
    explicit SyncRestorationRunner(std::unique_ptr<FactoryT> factory)
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

        model_path_ = args.modelPath;
        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(args.modelPath, io);

        if (!dxapp::minversionforRTandCompiler(&ie)) {
            std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                         "compatible with the version of the runtime. Please compile the model again."
                      << std::endl;
            return -1;
        }

        auto input_shape = ie.GetInputs().front().shape();
        int input_height, input_width;
        parseInputShape(input_shape, input_width, input_height);

        // Detect input layout and channels
        bool is_nhwc = isInputNHWC(input_shape);
        int input_channels = 1;
        if (input_shape.size() >= 4) {
            input_channels = static_cast<int>(is_nhwc ? input_shape[3] : input_shape[1]);
        }

        // Detect if model expects float input (e.g. Zero-DCE)
        bool is_float_input = (ie.GetInputs().front().type() == dxrt::DataType::FLOAT);

        // Load model configuration if provided
        if (!args.configPath.empty()) {
            dxapp::ModelConfig config(args.configPath);
            factory_->loadConfig(config);
        }

        auto preprocessor = factory_->createPreprocessor(input_width, input_height);
        auto postprocessor = factory_->createPostprocessor(input_width, input_height);
        auto visualizer = factory_->createVisualizer();

        std::cout << "[INFO] Model loaded: " << args.modelPath << std::endl;
        std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height << std::endl;
        std::cout << std::endl;

        SyncProfilingMetrics metrics;
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
            if (args.cameraIndex >= 0) {
                source_info = "Camera index: " + std::to_string(args.cameraIndex);
            } else if (!args.rtspUrl.empty()) {
                source_info = "RTSP URL: " + args.rtspUrl;
            } else {
                source_info = "Video file: " + args.videoFile;
                std::cout << "loopTest is set to 1 when a video file is provided." << std::endl;
                loopTest = 1;
            }

            if (args.verbose) {
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

            if (args.saveMode) {
                std::string run_name;
                if (args.cameraIndex >= 0) run_name = "camera" + std::to_string(args.cameraIndex);
                else if (!args.rtspUrl.empty()) run_name = "rtsp";
                else run_name = fs::path(args.videoFile).stem().string();
                std::string input_src = buildInputSourceString(
                    args.imageFilePath, args.videoFile, args.cameraIndex, args.rtspUrl);
                run_dir = makeRunDir(args.saveDir, factory_->getModelName() + "_sync",
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

        std::cout << "[INFO] Starting inference..." << std::endl;
        if (args.no_display) {
            std::cout << "Processing... Only FPS will be displayed." << std::endl;
        }

        cv::Mat display_image(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3);

        // Image mode: create run_dir when saving
        if (is_image && args.saveMode) {
            std::string run_kind = fs::is_directory(args.imageFilePath) ? "image-dir" : "image";
            std::string run_name = fs::path(args.imageFilePath).filename().string();
            std::string input_src = buildInputSourceString(
                args.imageFilePath, args.videoFile, args.cameraIndex, args.rtspUrl);
            run_dir = makeRunDir(args.saveDir, factory_->getModelName() + "_sync",
                                 run_kind, run_name);
            fs::create_directories(run_dir);
            writeRunInfo(run_dir, argv[0], args.modelPath, input_src);
        }

        std::string dumpTensorsBaseDir;
        if (args.dumpTensors && !is_image && !run_dir.empty()) {
            dumpTensorsBaseDir = run_dir + "/dump_tensors";
            fs::create_directories(dumpTensorsBaseDir);
            if (args.verbose) {
                std::cout << "[INFO] Dumping tensors to: " << dumpTensorsBaseDir << std::endl;
            }
        }

        auto s_time = std::chrono::high_resolution_clock::now();

        if (is_image) {
            processImageFrames(imageFiles, loopTest, display_image,
                               ie, *preprocessor, *postprocessor, *visualizer, metrics,
                               processCount, writer, args.no_display, args.saveMode,
                               input_channels, is_float_input, is_nhwc,
                               run_dir, args.dumpTensors);
        } else {
            processVideoFrames(video, display_image,
                               ie, *preprocessor, *postprocessor, *visualizer, metrics,
                               processCount, writer, args.no_display, args.saveMode,
                               input_channels, is_float_input, is_nhwc,
                               loopTest, args.videoFile,
                               dumpTensorsBaseDir);
        }

        auto e_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(e_time - s_time).count();

        if (g_interrupted().load()) {
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

        // DXAPP_VERIFY numerical verification
        verify::dumpVerifyJson(std::vector<RestorationResult>{}, model_path_, "restoration",
                              0, 0);

        printPerformanceSummary(metrics, processCount, total_time, !args.no_display, args.saveMode);

        DXRT_TRY_CATCH_END
        return 0;
    }

private:
    std::unique_ptr<FactoryT> factory_;
    std::string model_path_;  // Stored for DXAPP_VERIFY

    /** Dump input image on postprocessing exception for debugging. */
    static void dumpInputOnError(const cv::Mat& image) {
        std::string errDir = "error_tensors";
        fs::create_directories(errDir);
        std::string errPath = errDir + "/exception_input.bin";
        writeInputTensor(errPath, image);
        std::cerr << "[DXAPP] [ER] Auto-dumped input to: " << errPath << std::endl;
    }

    CommandLineArgs parseCommandLine(int argc, char* argv[]) {
        CommandLineArgs args;
        std::string app_name = factory_->getModelName() + " Image Restoration Sync Example";
        cxxopts::Options options(app_name, app_name + " application usage ");
        options.add_options()
            ("m, model_path", "restoration model file (.dxnn, required)",
             cxxopts::value<std::string>(args.modelPath))
            ("i, image_path", "input image file path or directory",
             cxxopts::value<std::string>(args.imageFilePath))
            ("v, video_path", "input video file path",
             cxxopts::value<std::string>(args.videoFile))
            ("c, camera_index", "camera device index",
             cxxopts::value<int>(args.cameraIndex))
            ("r, rtsp_url", "RTSP stream URL",
             cxxopts::value<std::string>(args.rtspUrl))
            ("s, save", "Save rendered output to disk",
             cxxopts::value<bool>(args.saveMode)->default_value("false"))
            ("save-dir", "Base directory for run outputs when using --save/--dump-tensors.",
             cxxopts::value<std::string>(args.saveDir)->default_value("artifacts/cpp_example"))
            ("dump-tensors", "(Debug) Always dump input/output tensors as .bin files.",
             cxxopts::value<bool>(args.dumpTensors)->default_value("false"))
            ("l, loop", "Number of inference iterations",
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
            dxapp::fatal_error("[ERROR] Model path is required. Use -m or --model_path option.");
        }
        int sourceCount = 0;
        if (!args.imageFilePath.empty()) sourceCount++;
        if (!args.videoFile.empty()) sourceCount++;
        if (args.cameraIndex >= 0) sourceCount++;
        if (!args.rtspUrl.empty()) sourceCount++;
        if (sourceCount != 1) {
            dxapp::fatal_error("[ERROR] Please specify exactly one input source.");
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
            if (loopTest == -1) loopTest = static_cast<int>(imageFiles.size());
        } else if (fs::is_regular_file(imageFilePath)) {
            imageFiles.push_back(imageFilePath);
            if (loopTest == -1) loopTest = 1;
        } else {
            dxapp::fatal_error("[ERROR] Invalid image path: ");
        }
        return {imageFiles, loopTest};
    }

    bool openVideoCapture(cv::VideoCapture& video, const CommandLineArgs& args) {
        if (args.cameraIndex >= 0) video.open(args.cameraIndex);
        else if (!args.rtspUrl.empty()) video.open(args.rtspUrl);
        else video.open(args.videoFile);
        return video.isOpened();
    }

    /** Compute output scale factors by probing the first tile. */
    std::pair<int,int> probeOutputScale(
        dxrt::InferenceEngine& ie, const cv::Mat& lr_gray, int tile_w, int tile_h,
        dxrt::TensorPtrs& probe_out) {
        cv::Mat probe_tile = lr_gray(cv::Rect(0, 0, tile_w, tile_h)).clone();
        probe_out = ie.Run(probe_tile.data, nullptr, nullptr);
        int out_tile_h = tile_h, out_tile_w = tile_w;
        if (!probe_out.empty()) {
            auto shape = probe_out[0]->shape();
            if (shape.size() == 4)      { out_tile_h = static_cast<int>(shape[2]); out_tile_w = static_cast<int>(shape[3]); }
            else if (shape.size() == 3) { out_tile_h = static_cast<int>(shape[1]); out_tile_w = static_cast<int>(shape[2]); }
            else if (shape.size() == 2) { out_tile_h = static_cast<int>(shape[0]); out_tile_w = static_cast<int>(shape[1]); }
        }
        return {std::max(1, out_tile_w / tile_w), std::max(1, out_tile_h / tile_h)};
    }

    /** Run tiled super-resolution and return side-by-side result canvas. */
    cv::Mat runSuperResolution(
        dxrt::InferenceEngine& ie, const cv::Mat& lr_bgr, const cv::Mat& lr_gray,
        int tile_w, int tile_h, int out_tile_w, int out_tile_h,
        dxrt::TensorPtrs& probe_out,
        double& t_inference_total, double& t_postprocess_total) {
        int scale_x = out_tile_w / tile_w, scale_y = out_tile_h / tile_h;
        int lr_w = lr_bgr.cols, lr_h = lr_bgr.rows;
        int out_w = lr_w * scale_x, out_h = lr_h * scale_y;
        cv::Mat sr_y(out_h, out_w, CV_8UC1, cv::Scalar(0));
        int tiles_done = 0, tiles_x = lr_w / tile_w, tiles_y = lr_h / tile_h;

        auto copy_tile_pixels = [&](const float* data, int dst_x, int dst_y) {
            for (int py = 0; py < out_tile_h; ++py)
                for (int px = 0; px < out_tile_w; ++px) {
                    float v = std::max(0.0f, std::min(1.0f, data[py * out_tile_w + px]));
                    sr_y.at<uchar>(dst_y + py, dst_x + px) = static_cast<uchar>(v * 255.0f + 0.5f);
                }
        };

        auto ti0 = std::chrono::high_resolution_clock::now();
        for (int ty = 0; ty < tiles_y; ++ty) {
            for (int tx = 0; tx < tiles_x; ++tx) {
                dxrt::TensorPtrs tile_out;
                if (ty == 0 && tx == 0) { tile_out = probe_out; }
                else {
                    cv::Mat tile = lr_gray(cv::Rect(tx*tile_w, ty*tile_h, tile_w, tile_h)).clone();
                    tile_out = ie.Run(tile.data, nullptr, nullptr);
                }
                if (tile_out.empty()) continue;
                const float* data = static_cast<const float*>(tile_out[0]->data());
                if (!data) continue;
                int dst_x = tx * out_tile_w, dst_y = ty * out_tile_h;
                copy_tile_pixels(data, dst_x, dst_y);
                ++tiles_done;
            }
        }
        t_inference_total = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - ti0).count();

        auto tp0 = std::chrono::high_resolution_clock::now();
        cv::Mat lr_ycrcb; cv::cvtColor(lr_bgr, lr_ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> ch; cv::split(lr_ycrcb, ch);
        cv::Mat cr_up, cb_up;
        cv::resize(ch[1], cr_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::resize(ch[2], cb_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat ycrcb_merged; cv::merge(std::vector<cv::Mat>{sr_y, cr_up, cb_up}, ycrcb_merged);
        cv::Mat sr_bgr; cv::cvtColor(ycrcb_merged, sr_bgr, cv::COLOR_YCrCb2BGR);
        t_postprocess_total = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tp0).count();

        cv::Mat lr_upscaled;
        cv::resize(lr_bgr, lr_upscaled, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat canvas(out_h, out_w * 2 + 4, CV_8UC3, cv::Scalar(0, 0, 0));
        lr_upscaled.copyTo(canvas(cv::Rect(0, 0, out_w, out_h)));
        sr_bgr.copyTo(canvas(cv::Rect(out_w + 4, 0, out_w, out_h)));
        cv::putText(canvas, cv::format("Bicubic (%dx%d)", lr_w, lr_h),
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 2);
        cv::putText(canvas,
                    cv::format("ESPCN x%d (%dx%d, %d tiles)", scale_x, out_w, out_h, tiles_done),
                    cv::Point(out_w + 14, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 100), 2);
        return canvas;
    }

    /** Run single-inference denoising/enhancement path.
     *  Returns: {false, Mat{}} -> postprocess error (caller should stop)
     *           {true, Mat{}} -> outputs empty (caller accumulates metrics and continues)
     *           {true, non-empty Mat} -> success
     */
    std::pair<bool, cv::Mat> runSingleInference(
        const cv::Mat& input_frame, cv::Mat& display_image,
        dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor,
        IPostprocessor<RestorationResult>& postprocessor,
        IVisualizer<RestorationResult>& visualizer,
        bool is_float_input, bool is_nhwc,
        double& t_preprocess, double& t_inference_total, double& t_postprocess_total) {
        // Preprocess using the ORIGINAL input frame so that PreprocessContext::source_image
        // (when stored by preprocessors like GrayscaleResizePreprocessor) contains the
        // full-resolution BGR image required by postprocessors (ESPCN color restoration).
        PreprocessContext ctx;
        cv::Mat preprocessed;
        auto tp0 = std::chrono::high_resolution_clock::now();
        preprocessor.process(input_frame, preprocessed, ctx);
        t_preprocess = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tp0).count();

        // Prepare a resized image for display (window) separately.
        dxapp::displayResize(input_frame, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);

        std::vector<float> float_buf;
        void* run_data = preprocessed.data;
        if (is_float_input && !preprocessed.empty()) {
            float_buf = convertToFloatBuffer(preprocessed, is_nhwc);
            run_data = float_buf.data();
        }

        auto ti0 = std::chrono::high_resolution_clock::now();
        auto outputs = ie.Run(run_data, nullptr, nullptr);
        t_inference_total = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - ti0).count();

        if (outputs.empty())
            return {true, cv::Mat{}};  // signal: no outputs, continue normally

        auto tpp0 = std::chrono::high_resolution_clock::now();
        std::vector<RestorationResult> results;
        try { results = postprocessor.process(outputs, ctx); }
        catch (const std::exception& e) {
            std::cerr << "[DXAPP] [ER] Postprocess error: " << e.what() << std::endl;
            // Auto-dump on exception
            dumpInputOnError(display_image);
            return {false, cv::Mat{}};  // signal: fatal error, stop processing
        }
        t_postprocess_total = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tpp0).count();
        return {true, visualizer.draw(display_image, results, ctx)};
    }

    bool processSingleFrame(
        const cv::Mat& input_frame, cv::Mat& display_image,
        dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor,
        IPostprocessor<RestorationResult>& postprocessor,
        IVisualizer<RestorationResult>& visualizer,
        SyncProfilingMetrics& metrics,
        cv::VideoWriter& writer, bool no_display, bool saveMode, double t_read,
        int input_channels = 1, bool is_float_input = false, bool is_nhwc = false,
        int frameIdx = 0,
        const std::string& dumpTensorsDir = "",
        bool dumpPerFrameDir = false) {

        if (input_frame.empty()) return false;

        int tile_w = preprocessor.getInputWidth();
        int tile_h = preprocessor.getInputHeight();

        cv::Mat result_frame;
        double t_preprocess = 0.0, t_inference_total = 0.0, t_postprocess_total = 0.0;

        bool is_sr = false;
        if (input_channels <= 1) {
            auto t0 = std::chrono::high_resolution_clock::now();
            const int TARGET_TILES_W = 20;
            int lr_w = tile_w * TARGET_TILES_W;
            int lr_h = static_cast<int>(std::round(
                static_cast<double>(lr_w) * input_frame.rows / input_frame.cols));
            lr_h = ((lr_h + tile_h - 1) / tile_h) * tile_h;
            if (lr_h <= 0) lr_h = tile_h * 10;

            cv::Mat lr_bgr; cv::resize(input_frame, lr_bgr, cv::Size(lr_w, lr_h));
            cv::Mat lr_gray; cv::cvtColor(lr_bgr, lr_gray, cv::COLOR_BGR2GRAY);

            dxrt::TensorPtrs probe_out;
            std::pair<int,int> scale_xy = probeOutputScale(ie, lr_gray, tile_w, tile_h, probe_out);
            int scale_x = scale_xy.first;
            int scale_y = scale_xy.second;
            t_preprocess = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            is_sr = (scale_x > 1 || scale_y > 1);

            if (is_sr) {
                int out_tile_w = tile_w * scale_x, out_tile_h = tile_h * scale_y;
                result_frame = runSuperResolution(ie, lr_bgr, lr_gray, tile_w, tile_h,
                    out_tile_w, out_tile_h, probe_out,
                    t_inference_total, t_postprocess_total);
                display_image = result_frame;
            }
        }

        if (!is_sr) {
            auto maybe = runSingleInference(input_frame, display_image, ie,
                preprocessor, postprocessor, visualizer,
                is_float_input, is_nhwc,
                t_preprocess, t_inference_total, t_postprocess_total);
            if (!maybe.first) return false;  // postprocess error
            if (maybe.second.empty()) {
                // outputs were empty: accumulate metrics and continue
                metrics.sum_read += t_read;
                metrics.sum_preprocess += t_preprocess;
                metrics.sum_inference += t_inference_total;
                metrics.infer_completed++;
                return true;
            }
            result_frame = std::move(maybe.second);
        }

        // Dump tensors if enabled
        if (!dumpTensorsDir.empty()) {
            std::string dumpDir = dumpTensorsDir;
            if (dumpPerFrameDir) {
                dumpDir = dumpTensorsDir + "/frame" + std::to_string(frameIdx);
            }
            // Dump input frame as .bin
            fs::create_directories(dumpDir);
            std::string input_path = dumpDir + "/input_frame.bin";
            std::ofstream ofs(input_path, std::ios::binary);
            if (ofs.is_open()) {
                ofs.write(static_cast<const char*>(static_cast<const void*>(input_frame.data)),
                          input_frame.total() * input_frame.elemSize());
                ofs.close();
            }
        }

        auto render_start = std::chrono::high_resolution_clock::now();
        bool quit_requested = renderAndDisplay_(result_frame, writer, no_display, saveMode);
        auto render_end = std::chrono::high_resolution_clock::now();
        double t_render = std::chrono::duration<double, std::milli>(render_end - render_start).count();

        metrics.sum_read += t_read;
        metrics.sum_preprocess += t_preprocess;
        metrics.sum_inference += t_inference_total;
        metrics.sum_postprocess += t_postprocess_total;
        metrics.sum_render += t_render;
        metrics.infer_completed++;

        return !quit_requested;
    }

    // Render result frame: save video, save image, display.
    // Returns true when the user requested quit (pressed 'q').
    bool renderAndDisplay_(const cv::Mat& result_frame, cv::VideoWriter& writer,
                           bool no_display, bool saveMode) const {
        bool quit_requested = false;
        if (result_frame.empty()) return false;

        if (saveMode) {
            if (result_frame.cols != static_cast<int>(SHOW_WINDOW_SIZE_W) ||
                result_frame.rows != static_cast<int>(SHOW_WINDOW_SIZE_H)) {
                cv::Mat write_frame;
                cv::resize(result_frame, write_frame,
                           cv::Size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H));
                writer << write_frame;
            } else {
                writer << result_frame;
            }
        }
        dxapp::saveDebugImage(result_frame);

            if (!no_display) {
                cv::imshow("Output", result_frame);
                if (dxapp::windowShouldClose("Output")) quit_requested = true;
            }
        return quit_requested;
    }

    void processImageFrames(
        const std::vector<std::string>& imageFiles, int loopTest,
        cv::Mat& display_image, dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor, IPostprocessor<RestorationResult>& postprocessor,
        IVisualizer<RestorationResult>& visualizer, SyncProfilingMetrics& metrics,
        int& processCount, cv::VideoWriter& writer, bool no_display, bool saveMode,
        int input_channels = 1, bool is_float_input = false, bool is_nhwc = false,
        const std::string& runDir = "", bool dumpEnabled = false) {
        for (int i = 0; i < loopTest; ++i) {
            std::string currentImagePath = imageFiles[i % imageFiles.size()];
            // Set per-image save path for this frame when saveMode is enabled
            if (!runDir.empty() && dumpEnabled) {
                std::string savePath = dxapp::buildPerImageSavePath(runDir, factory_->getModelName() + "_sync", currentImagePath, i);
                setenv("DXAPP_SAVE_IMAGE", savePath.c_str(), 1);
            }
            auto tr0 = std::chrono::high_resolution_clock::now();
            cv::Mat img = cv::imread(currentImagePath);
            auto tr1 = std::chrono::high_resolution_clock::now();
            double t_read = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
            if (img.empty()) continue;
            // Per-frame dump directory
            std::string frameDumpPath;
            if (dumpEnabled && !runDir.empty()) {
                std::string fname = fs::path(currentImagePath).filename().string();
                frameDumpPath = runDir + "/" + fname + "/dump_tensors";
            }
            if (!processSingleFrame(img, display_image, ie, preprocessor, postprocessor,
                                    visualizer, metrics, writer, no_display, saveMode, t_read,
                                    input_channels, is_float_input, is_nhwc,
                                    i, frameDumpPath, false)) break;
            processCount++;
            if (!no_display) {
                while (!dxapp::windowShouldClose("Output")) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        }
    }

    void processVideoFrames(
        cv::VideoCapture& video, cv::Mat& display_image, dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor, IPostprocessor<RestorationResult>& postprocessor,
        IVisualizer<RestorationResult>& visualizer, SyncProfilingMetrics& metrics,
        int& processCount, cv::VideoWriter& writer, bool no_display, bool saveMode,
        int input_channels = 1, bool is_float_input = false, bool is_nhwc = false,
        int loopTest = 1, const std::string& videoFile = "",
        const std::string& dumpTensorsBaseDir = "") {
        for (int loop_idx = 0; loop_idx < loopTest && !g_interrupted().load(); ++loop_idx) {
            if (loopTest > 1) {
                if (verbose_) {
                    std::cout << "\n" << std::string(50, '=') << std::endl;
                    std::cout << "[INFO] Loop " << (loop_idx + 1) << "/" << loopTest << std::endl;
                    std::cout << std::string(50, '=') << std::endl;
                }
            }
            double t_read = 0.0;
            auto readFrame = [&video, &t_read](cv::Mat& f) {
                auto t0 = std::chrono::high_resolution_clock::now();
                video >> f;
                auto t1 = std::chrono::high_resolution_clock::now();
                t_read = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return !f.empty();
            };
            cv::Mat frame;
            while (!g_interrupted().load() && readFrame(frame)) {
                if (!processSingleFrame(frame, display_image, ie, preprocessor, postprocessor,
                                        visualizer, metrics, writer, no_display, saveMode, t_read,
                                        input_channels, is_float_input, is_nhwc,
                                    processCount, dumpTensorsBaseDir, true)) break;
                processCount++;
            }
            // Reopen video for next loop
            if (loop_idx + 1 >= loopTest || videoFile.empty()) continue;
            video.release();
            video.open(videoFile);
            if (!video.isOpened()) {
                std::cerr << "[ERROR] Failed to reopen video for loop " << (loop_idx + 2) << std::endl;
                break;
            }
        }
    }

    void printPerformanceSummary(const SyncProfilingMetrics& metrics, int total_frames,
                                double total_time_sec, bool display_on, bool save_on = false) {
        if (metrics.infer_completed == 0) return;

        double avg_read = metrics.sum_read / metrics.infer_completed;
        double avg_pre = metrics.sum_preprocess / metrics.infer_completed;
        double avg_inf = metrics.sum_inference / metrics.infer_completed;
        double avg_post = metrics.sum_postprocess / metrics.infer_completed;

        double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
        double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
        double infer_fps = avg_inf > 0 ? 1000.0 / avg_inf : 0.0;
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
                  << std::setprecision(1) << infer_fps << " FPS" << std::endl;
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
        if (save_on) {
            double avg_save = metrics.sum_save / metrics.infer_completed;
            double save_fps = avg_save > 0 ? 1000.0 / avg_save : 0.0;
            std::cout << " " << std::left << std::setw(15) << "Save" << std::right << std::setw(8)
                      << std::fixed << std::setprecision(2) << avg_save << " ms     " << std::setw(6)
                      << std::setprecision(1) << save_fps << " FPS" << std::endl;
        }
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << " " << std::left << std::setw(19) << "Total Frames"
                  << " :    " << total_frames << std::endl;
        std::cout << " " << std::left << std::setw(19) << "Total Time"
                  << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s" << std::endl;
        double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;
        std::cout << " " << std::left << std::setw(19) << "Overall FPS"
                  << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS" << std::endl;
        std::cout << "==================================================" << std::endl;
    }
};

}  // namespace dxapp

#endif  // RESTORATION_RUNNER_HPP
