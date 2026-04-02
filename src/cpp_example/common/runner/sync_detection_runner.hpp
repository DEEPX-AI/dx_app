/**
 * @file sync_detection_runner.hpp
 * @brief Synchronous inference runner using factory pattern
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Provides a generic runner that accepts any factory implementation.
 */

#ifndef SYNC_RUNNER_HPP
#define SYNC_RUNNER_HPP

#include <dxrt/dxrt_api.h>
#include <chrono>
#include <csignal>
#include <thread>
#include <cxxopts.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <tuple>
#include <vector>

#include "common/base/i_factory.hpp"
#include "common/config/model_config.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/verify_serialize.hpp"

namespace dxapp {

constexpr size_t SHOW_WINDOW_SIZE_W = 960;
constexpr size_t SHOW_WINDOW_SIZE_H = 640;

// Profiling metrics structure
struct SyncProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    double sum_postprocess = 0.0;
    double sum_render = 0.0;
    double sum_save = 0.0;
    int infer_completed = 0;
};

// Command line arguments structure
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

// --- Shared helper functions for sync runners ---

/** Zero-padded index formatting helper. */
inline std::string formatIndex(int value, int width) {
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(width) << value;
    return oss.str();
}

/** Build dump-tensors directory path for a single image frame. */
inline std::string buildFrameDumpPath(
    bool dumpEnabled, const std::string& runDir,
    bool multi_loop, int loop_idx, bool is_dir_input,
    const std::string& imagePath) {
    if (!dumpEnabled || runDir.empty()) return "";
    std::string dumpLoopDir = multi_loop
        ? (runDir + "/loop" + formatIndex(loop_idx + 1, 3))
        : runDir;
    std::string fname = fs::path(imagePath).filename().string();
    return is_dir_input
        ? (dumpLoopDir + "/" + fname + "/dump_tensors")
        : (dumpLoopDir + "/dump_tensors");
}

/** Build save-image path for a single image frame (first loop only). */
inline std::string buildSaveImagePath(
    bool should_save, const std::string& runDir,
    bool is_dir_input, bool multi_loop, int images_per_loop,
    int img_idx, const std::string& imagePath) {
    if (!should_save || runDir.empty()) return "";
    std::string loopDir = multi_loop ? (runDir + "/loop001") : runDir;
    if (is_dir_input) {
        std::string fname = fs::path(imagePath).filename().string();
        return loopDir + "/" + fname + "/output.jpg";
    }
    if (images_per_loop > 1) {
        return loopDir + "/output_" + formatIndex(img_idx + 1, 4) + ".jpg";
    }
    return loopDir + "/output.jpg";
}

/** Render results, save output, and display. Returns {t_render, t_save, quit_requested}. */
template <typename ResultT, typename VisualizerT>
inline std::tuple<double, double, bool> renderSaveDisplay(
    const cv::Mat& display_image,
    const std::vector<ResultT>& results,
    const PreprocessContext& ctx,
    VisualizerT& visualizer,
    cv::VideoWriter& writer,
    bool no_display, bool saveMode,
    const std::string& saveImagePath = "",
    bool verbose = false) {

    double t_render = 0.0;
    double t_save = 0.0;
    bool quit_requested = false;
    auto render_start = std::chrono::high_resolution_clock::now();
    cv::Mat result_frame = display_image.clone();
    result_frame = visualizer.draw(result_frame, results, ctx);
    auto render_end = std::chrono::high_resolution_clock::now();
    t_render = std::chrono::duration<double, std::milli>(render_end - render_start).count();

    if (result_frame.empty()) return {t_render, t_save, quit_requested};

    if (saveMode) {
        auto save_start = std::chrono::high_resolution_clock::now();
        if (!saveImagePath.empty()) {
            fs::create_directories(fs::path(saveImagePath).parent_path());
            cv::imwrite(saveImagePath, result_frame);
            if (verbose) {
                std::cout << "\n[INFO] Saved output image: "
                      << fs::absolute(saveImagePath).string() << std::endl;
            }
        } else if (writer.isOpened()) {
            writer << result_frame;
        }
        auto save_end = std::chrono::high_resolution_clock::now();
        t_save = std::chrono::duration<double, std::milli>(save_end - save_start).count();
    }
    dxapp::saveDebugImage(result_frame);
    if (!no_display) {
        cv::imshow("Output", result_frame);
        if (dxapp::windowShouldClose("Output")) {
            quit_requested = true;
        }
    }
    return {t_render, t_save, quit_requested};
}

/**
 * @brief Generic synchronous runner for detection-based models
 * @tparam FactoryT The factory type (must derive from IDetectionFactory)
 */
template <typename FactoryT>
class SyncDetectionRunner {
    bool verbose_ = false;

public:
    explicit SyncDetectionRunner(std::unique_ptr<FactoryT> factory)
        : factory_(std::move(factory)) {}

    int run(int argc, char* argv[]) {
        DXRT_TRY_CATCH_BEGIN
        installSignalHandlers();
        int processCount = 0;

        // Parse command line arguments
        CommandLineArgs args = parseCommandLine(argc, argv);
        verbose_ = args.verbose;
        validateArguments(args);

        // Handle image file or directory
        std::vector<std::string> imageFiles;
        bool is_image = !args.imageFilePath.empty();
        bool is_dir_input = is_image && fs::is_directory(args.imageFilePath);
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

        // Detect float input model (e.g., CenterNet_ResNet18) and layout
        bool is_float_input = (ie.GetInputs().front().type() == dxrt::DataType::FLOAT);
        bool is_nhwc = isInputNHWC(input_shape);

        // Create processors and visualizer using factory
        // Load model configuration if provided
        if (!args.configPath.empty()) {
            dxapp::ModelConfig config(args.configPath);
            factory_->loadConfig(config);
        }

        auto preprocessor = factory_->createPreprocessor(input_width, input_height);
        auto postprocessor = factory_->createPostprocessor(input_width, input_height, ie.IsOrtConfigured());
        auto visualizer = factory_->createVisualizer();
        std::string scriptTag = factory_->getModelName() + "_sync";

        std::cout << "[INFO] Model loaded: " << args.modelPath << std::endl;
        std::cout << "[INFO] Model input size (WxH): " << input_width << "x" << input_height << std::endl;
        std::cout << std::endl;

        if (is_image) {
            auto images_per_loop = static_cast<int>(imageFiles.size());
            int user_loop_count = (args.loopTest == -1) ? 1 : args.loopTest;
            if (is_dir_input) {
                if (args.verbose) {
                    std::cout << "[INFO] Found " << images_per_loop << " images in directory: "
                          << args.imageFilePath << std::endl;
                }
            }
            if (args.verbose) {
                std::cout << "[INFO] Image input (" << images_per_loop << " files)" << std::endl;
                std::cout << "[INFO] Images per loop: " << images_per_loop << std::endl;
                std::cout << "[INFO] Loop count: " << user_loop_count << std::endl;
                std::cout << std::endl;
            }
        }

        // Allocate input buffer
        std::vector<uint8_t> input_buffer(ie.GetInputSize());
        SyncProfilingMetrics metrics;

        cv::VideoCapture video;
        cv::VideoWriter writer;

        // --- Create run directory for save/dump-tensor output ---
        std::string runDir;
        if (args.saveMode || args.dumpTensors) {
            std::string runKind;
            std::string runName;
            if (is_image) {
                runKind = is_dir_input ? "image-dir" : "image";
                runName = fs::path(args.imageFilePath).filename().string();
            } else if (args.cameraIndex >= 0) {
                runKind = "stream";
                runName = "camera" + std::to_string(args.cameraIndex);
            } else if (!args.rtspUrl.empty()) {
                runKind = "stream";
                runName = "rtsp";
            } else {
                runKind = "stream";
                runName = fs::path(args.videoFile).stem().string();
            }
            runDir = makeRunDir(args.saveDir, scriptTag, runKind, runName);
            fs::create_directories(runDir);
            std::string inputSource = buildInputSourceString(
                args.imageFilePath, args.videoFile, args.cameraIndex, args.rtspUrl);
            writeRunInfo(runDir, argv[0], args.modelPath, inputSource);
        }

        std::string dumpTensorsBaseDir;
        if (args.dumpTensors && !is_image) {
            dumpTensorsBaseDir = runDir + "/dump_tensors";
            fs::create_directories(dumpTensorsBaseDir);
            if (args.verbose) {
                std::cout << "[INFO] Dumping tensors to: " << dumpTensorsBaseDir << std::endl;
            }
        }

        // --- Video source setup ---
        int vid_frame_width = 0;
        int vid_frame_height = 0;
        int vid_total_frames = 0;
        double vid_fps = 0.0;
        if (!is_image) {
            if (!openVideoCapture(video, args)) {
                std::cerr << "[ERROR] Failed to open input source." << std::endl;
                return -1;
            }

            vid_frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
            vid_frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
            vid_fps = video.get(cv::CAP_PROP_FPS);
            vid_total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

            std::string source_info;
            if (args.cameraIndex >= 0) {
                source_info = "Camera index: " + std::to_string(args.cameraIndex);
            } else if (!args.rtspUrl.empty()) {
                source_info = "RTSP URL: " + args.rtspUrl;
            } else {
                source_info = "Video file: " + args.videoFile;
            }
            if (args.verbose) {
                std::cout << "[INFO] " << source_info << std::endl;
                std::cout << "[INFO] Input source resolution (WxH): "
                << vid_frame_width << "x" << vid_frame_height << std::endl;
                std::cout << "[INFO] Input source FPS: " << std::fixed
                << std::setprecision(2) << vid_fps << std::endl;
            }
            if (!args.videoFile.empty()) {
                if (args.verbose) {
                    std::cout << "[INFO] Total frames: " << vid_total_frames << std::endl;
                }
            }
            if (loopTest > 1) {
                if (args.verbose) {
                    std::cout << "[INFO] Loop count: " << loopTest << std::endl;
                }
            }
            std::cout << std::endl;
        }

        std::cout << "[INFO] Starting inference..." << std::endl;
        if (args.no_display) {
            std::cout << "Processing... Only FPS will be displayed." << std::endl;
        }

        cv::Mat display_image(SHOW_WINDOW_SIZE_H, SHOW_WINDOW_SIZE_W, CV_8UC3);
        cv::Mat preprocessed_image(input_height, input_width, CV_8UC3, input_buffer.data());
        auto s_time = std::chrono::high_resolution_clock::now();

        if (is_image) {
            int user_loop_count = (args.loopTest == -1) ? 1 : args.loopTest;
            processImageFrames(imageFiles, is_dir_input, user_loop_count,
                               display_image, preprocessed_image,
                               ie, *preprocessor, *postprocessor, *visualizer, metrics,
                               processCount, writer, args.no_display, args.saveMode,
                               is_float_input, is_nhwc, runDir, args.dumpTensors);
        } else {
            processVideoFrames(video, display_image, preprocessed_image,
                               ie, *preprocessor, *postprocessor, *visualizer, metrics,
                               processCount, writer, args, loopTest,
                               is_float_input, is_nhwc, runDir,
                               dumpTensorsBaseDir, vid_fps);
        }

        auto e_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(e_time - s_time).count();

        if (g_interrupted().load()) {
            std::cout << "[INFO] Interrupted by user (Ctrl+C)" << std::endl;
        }
        if (writer.isOpened()) writer.release();

        // Performance summary
        if (is_image) {
            int user_loop_count = (args.loopTest == -1) ? 1 : args.loopTest;
            if (user_loop_count > 1) {
                std::cout << "\n[INFO] Average performance over "
                          << user_loop_count << " loops" << std::endl;
            }
        } else if (loopTest > 1) {
            std::cout << "\n[INFO] Average performance over "
                      << loopTest << " loops" << std::endl;
        }
        printPerformanceSummary(metrics, processCount, total_time,
                                !args.no_display || args.saveMode, args.saveMode);

        DXRT_TRY_CATCH_END
        return 0;
    }

private:
    std::unique_ptr<FactoryT> factory_;
    std::string model_path_;

    /** Setup or rewind video writer/capture at the start of each video loop iteration. */
    void setupOrRewindVideo(int loop_idx, int loopTest, const CommandLineArgs& args,
                            cv::VideoCapture& video, cv::VideoWriter& writer,
                            const std::string& runDir, double vid_fps,
                            std::string& videoSavePathResolved) {
        if (loop_idx == 0 && args.saveMode && !runDir.empty()) {
            std::string loopSaveDir = (loopTest > 1) ? runDir + "/loop001" : runDir;
            cv::Size save_size(SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
            writer = initVideoWriter(loopSaveDir, vid_fps, save_size, videoSavePathResolved);
            if (!writer.isOpened()) {
                std::cerr << "[ERROR] Failed to open video writer." << std::endl;
            }
        } else if (loop_idx > 0 && !args.videoFile.empty()) {
            video.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
    }

    /** Finalize video writer after first loop and log the saved path. */
    static void finalizeVideoWriter(int loop_idx, bool saveMode,
                                    cv::VideoWriter& writer,
                                    const std::string& videoSavePathResolved,
                                    bool verbose = false) {
        if (loop_idx == 0 && saveMode && writer.isOpened()) {
            writer.release();
            if (!videoSavePathResolved.empty()) {
                if (verbose) {
                    std::cout << "\n[INFO] Saved output video: " << videoSavePathResolved << std::endl;
                }
            }
        }
    }

    // --- Command line parsing (matching Legacy format) ---

    CommandLineArgs parseCommandLine(int argc, char* argv[]) {
        CommandLineArgs args;
        std::string app_name = factory_->getModelName() + " Post-Processing Sync Example";
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
            ("s, save", "Save rendered output (image/stream) to disk. "
                        "When used with --loop > 1, only the first loop is saved.",
             cxxopts::value<bool>(args.saveMode)->default_value("false"))
            ("save-dir", "Base directory for run outputs when using --save/--dump-tensors.",
             cxxopts::value<std::string>(args.saveDir)->default_value("artifacts/cpp_example"))
            ("dump-tensors", "(Debug) Always dump input/output tensors as .bin files.",
             cxxopts::value<bool>(args.dumpTensors)->default_value("false"))
            ("l, loop", "Repeat inference N times (valid for --image/--video only).",
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

    // --- Image path processing ---

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

    // --- Video capture ---

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

    // --- Postprocessing with exception handling (matching Legacy) ---

    /** Print detection results to stdout for pipeline parsing. */
    static void printDetectionResults(
        const std::vector<DetectionResult>& detections, int frame_w, int frame_h, bool verbose = false) {
        for (const auto& det : detections) {
            if (det.box.size() < 4) continue;
            std::string dname = det.class_name;
            for (auto& ch : dname) if (ch == ' ') ch = '_';
            if (verbose) {
                std::cout << "[DET] " << dname
                          << " " << det.confidence
                          << " " << det.box[0] << " " << det.box[1]
                          << " " << det.box[2] << " " << det.box[3]
                          << " " << frame_w << " " << frame_h
                          << std::endl;
            }
        }
    }

    // --- Frame processing ---

    bool processSingleFrame(
        const cv::Mat& input_frame, cv::Mat& display_image,
        dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor,
        IPostprocessor<DetectionResult>& postprocessor,
        IVisualizer<DetectionResult>& visualizer,
        SyncProfilingMetrics& metrics,
        cv::VideoWriter& writer, bool no_display, bool saveMode, double t_read,
        bool is_float_input = false, bool is_nhwc = false,
        const std::string& saveImagePath = "",
        int frameIdx = 0,
        const std::string& dumpTensorsDir = "",
        bool dumpPerFrameDir = false) {

        if (input_frame.empty()) {
            LOG_ERROR("Empty input frame");
            return false;
        }

        // Preprocess using the original input frame so ctx.original_width/height
        // reflect the true source image size. Resize for display afterwards.
        auto t0 = std::chrono::high_resolution_clock::now();
        PreprocessContext ctx;
        cv::Mat preprocessed;
        preprocessor.process(input_frame, preprocessed, ctx);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Prepare display image (window-sized) from original frame
        dxapp::displayResize(input_frame, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);

        // If model expects float input, convert uint8 to float32
        std::vector<float> float_buf;
        void* run_data = preprocessed.data;
        if (is_float_input && !preprocessed.empty()) {
            float_buf = convertToFloatBuffer(preprocessed, is_nhwc);
            run_data = float_buf.data();
        }

        // Inference (Legacy API: ie.Run())
        auto outputs = ie.Run(run_data, nullptr, nullptr);
        auto t2 = std::chrono::high_resolution_clock::now();
        double t_inference = std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (outputs.empty()) {
            metrics.sum_read += t_read;
            metrics.sum_preprocess += t_preprocess;
            metrics.sum_inference += t_inference;
            metrics.infer_completed++;
            return true;
        }

        // Dump tensors if requested (normal path)
        if (!dumpTensorsDir.empty()) {
            std::string actualDumpDir = frameDumpDir(dumpTensorsDir, frameIdx, dumpPerFrameDir);
            dumpTensorsToFiles(actualDumpDir, preprocessed, outputs);
            if (verbose_) {
                std::cout << "\n[INFO] Saved input & output tensors: "
                          << fs::absolute(actualDumpDir).string() << std::endl;
            }
        }

        // Postprocess
        std::vector<DetectionResult> detections;
        auto t_post_start = std::chrono::high_resolution_clock::now();
        try {
            detections = postprocessor.process(outputs, ctx);
        } catch (const std::exception& e) {
            LOG_ERROR("Postprocessing error: " << e.what());
            // Auto-dump tensors on exception for debugging
            if (dumpTensorsDir.empty()) {
                static const std::string kErrorDumpDir = "error_tensors";
                dumpTensorsToFiles(kErrorDumpDir, preprocessed, outputs);
                if (verbose_) {
                    std::cout << "[INFO] Auto-dumped tensors on exception to: "
                              << fs::absolute(kErrorDumpDir).string() << std::endl;
                }
            }
            dxapp::saveDebugImage(display_image);
            metrics.sum_read += t_read;
            metrics.sum_preprocess += t_preprocess;
            metrics.sum_inference += t_inference;
            metrics.infer_completed++;
            return true;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double t_postprocess = std::chrono::duration<double, std::milli>(t3 - t_post_start).count();

        // Print detection results to stdout for pipeline parsing
        printDetectionResults(detections, display_image.cols, display_image.rows, verbose_);


        // --- Numerical verification dump (DXAPP_VERIFY=1) ---
        verify::dumpVerifyJson(detections, model_path_,
            "object_detection", display_image.rows, display_image.cols);

        // Render
        double t_render = 0.0;
        double t_save = 0.0;
        bool quit_requested = false;
        if (!no_display || saveMode || std::getenv("DXAPP_SAVE_IMAGE")) {
            // Draw/save on the original input frame so detection boxes (in original
            // coordinates) map correctly. display_image is a resized window copy.
            std::tie(t_render, t_save, quit_requested) = renderSaveDisplay(
                input_frame, detections, ctx, visualizer, writer,
                no_display, saveMode, saveImagePath, verbose_);
        }

        // Update metrics
        metrics.sum_read += t_read;
        metrics.sum_preprocess += t_preprocess;
        metrics.sum_inference += t_inference;
        metrics.sum_postprocess += t_postprocess;
        metrics.sum_render += t_render;
        metrics.sum_save += t_save;
        metrics.infer_completed++;

        return !quit_requested;
    }

    void processImageFrames(
        const std::vector<std::string>& imageFiles, bool is_dir_input,
        int user_loop_count, cv::Mat& display_image, const cv::Mat& preprocessed_image,
        dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor,
        IPostprocessor<DetectionResult>& postprocessor,
        IVisualizer<DetectionResult>& visualizer,
        SyncProfilingMetrics& metrics,
        int& processCount, cv::VideoWriter& writer, bool no_display, bool saveMode,
        bool is_float_input, bool is_nhwc,
        const std::string& runDir, bool dumpEnabled) {

        auto images_per_loop = static_cast<int>(imageFiles.size());
        bool multi_loop = user_loop_count > 1;

        for (int loop_idx = 0; loop_idx < user_loop_count && !g_interrupted().load(); ++loop_idx) {
            if (multi_loop) {
                if (verbose_) {
                    std::cout << "\n" << std::string(50, '=') << std::endl;
                    std::cout << "[INFO] Loop " << (loop_idx + 1) << "/" << user_loop_count << std::endl;
                    std::cout << std::string(50, '=') << std::endl;
                }
            }

            for (int img_idx = 0; img_idx < images_per_loop && !g_interrupted().load(); ++img_idx) {
                std::string currentImagePath = imageFiles[img_idx];

                auto tr0 = std::chrono::high_resolution_clock::now();
                cv::Mat img = cv::imread(currentImagePath);
                auto tr1 = std::chrono::high_resolution_clock::now();
                double t_read = std::chrono::duration<double, std::milli>(tr1 - tr0).count();

                if (img.empty()) {
                    std::cerr << "[ERROR] Failed to read image: " << currentImagePath << std::endl;
                    continue;
                }

                if (is_dir_input || multi_loop) {
                    std::string filename = fs::path(currentImagePath).filename().string();
                    if (verbose_) {
                        std::cout << "\n[INFO] Image " << (img_idx + 1) << "/" << images_per_loop
                              << ": " << filename << std::endl;
                    }
                }

                if (loop_idx == 0 || (is_dir_input && !multi_loop)) {
                    if (verbose_) {
                        std::cout << "[INFO] Input image: " << currentImagePath << std::endl;
                        std::cout << "[INFO] Image resolution (WxH): "
                        << img.cols << "x" << img.rows << std::endl;
                    }
                }

                bool should_save = saveMode && loop_idx == 0;

                // Per-frame dump directory
                std::string frameDumpPath = buildFrameDumpPath(
                    dumpEnabled, runDir, multi_loop, loop_idx, is_dir_input, currentImagePath);

                // Save image path
                std::string saveImagePath = buildSaveImagePath(
                    should_save, runDir, is_dir_input, multi_loop, images_per_loop,
                    img_idx, currentImagePath);

                int flat_idx = loop_idx * images_per_loop + img_idx;
                if (!processSingleFrame(img, display_image, ie, preprocessor, postprocessor,
                                        visualizer, metrics, writer, no_display, should_save, t_read,
                                        is_float_input, is_nhwc, saveImagePath, flat_idx,
                                        frameDumpPath, false)) {
                    return;
                }
                processCount++;
                // In image mode, block after showing the frame until user closes
                // the window or presses q/ESC — matches Python behavior.
                if (!no_display) {
                    while (!dxapp::windowShouldClose("Output")) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
            }
        }
    }

    void processVideoFrames(
        cv::VideoCapture& video, cv::Mat& display_image, const cv::Mat& /*preprocessed_image*/,
        dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor,
        IPostprocessor<DetectionResult>& postprocessor,
        IVisualizer<DetectionResult>& visualizer,
        SyncProfilingMetrics& metrics,
        int& processCount, cv::VideoWriter& writer,
        const CommandLineArgs& args, int loopTest,
        bool is_float_input, bool is_nhwc,
        const std::string& runDir, const std::string& dumpTensorsBaseDir,
        double vid_fps) {

        std::string videoSavePathResolved;

        for (int loop_idx = 0; loop_idx < loopTest && !g_interrupted().load(); ++loop_idx) {
            if (loopTest > 1) {
                if (verbose_) {
                    std::cout << "\n" << std::string(50, '=') << std::endl;
                    std::cout << "[INFO] Loop " << (loop_idx + 1) << "/" << loopTest << std::endl;
                    std::cout << std::string(50, '=') << std::endl;
                }
            }

            // Setup VideoWriter on first loop or rewind for subsequent loops
            setupOrRewindVideo(loop_idx, loopTest, args, video, writer,
                               runDir, vid_fps, videoSavePathResolved);

            bool saveThisLoop = args.saveMode && (loop_idx == 0);
            int frameIdx = 0;
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
                                        visualizer, metrics, writer, args.no_display, saveThisLoop, t_read,
                                        is_float_input, is_nhwc, "", frameIdx,
                                        dumpTensorsBaseDir, true)) {
                    break;
                }
                processCount++;
                frameIdx++;
            }

            finalizeVideoWriter(loop_idx, args.saveMode, writer, videoSavePathResolved, args.verbose);
        }
    }

    // --- Performance summary (matching Legacy format exactly) ---

    void printPerformanceSummary(const SyncProfilingMetrics& metrics, int total_frames,
                                double total_time_sec, bool display_on,
                                bool save_on = false) {
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
        if (save_on && metrics.sum_save > 0) {
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
                  << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
                  << std::endl;

        double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;
        std::cout << " " << std::left << std::setw(19) << "Overall FPS"
                  << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
                  << std::endl;
        std::cout << "==================================================" << std::endl;
    }
};

}  // namespace dxapp

#endif  // SYNC_RUNNER_HPP
