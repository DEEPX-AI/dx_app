/**
 * @file obb_runner.hpp
 * @brief Synchronous OBB (Oriented Bounding Box) runner using factory pattern
 *
 * Provides a generic runner that accepts any IOBBFactory implementation.
 */

#ifndef OBB_RUNNER_HPP
#define OBB_RUNNER_HPP

#include <dxrt/dxrt_api.h>
#include <chrono>
#include <cxxopts.hpp>
#include <thread>
#include <experimental/filesystem>
#include <iomanip>
#include <cstdlib>
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

/**
 * @brief Generic synchronous runner for OBB detection models
 * @tparam FactoryT The factory type (must derive from IOBBFactory)
 */
template <typename FactoryT>
class SyncOBBRunner {
    bool verbose_ = false;

public:
    explicit SyncOBBRunner(std::unique_ptr<FactoryT> factory)
        : factory_(std::move(factory)) {}

    int run(int argc, char* argv[]) {
        DXRT_TRY_CATCH_BEGIN
        installSignalHandlers();
        int processCount = 0;

        CommandLineArgs args = parseCommandLine(argc, argv);
        verbose_ = args.verbose;
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

        // Initialize inference engine (Legacy API)
        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(args.modelPath, io);
        model_path_ = args.modelPath;

        if (!dxapp::minversionforRTandCompiler(&ie)) {
            std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                         "compatible with the version of the runtime. Please compile the model again."
                      << std::endl;
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
        auto postprocessor = factory_->createPostprocessor(input_width, input_height, ie.IsOrtConfigured());
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
                               run_dir, args.dumpTensors);
        } else {
            processVideoFrames(video, display_image,
                               ie, *preprocessor, *postprocessor, *visualizer, metrics,
                               processCount, writer, args.no_display, args.saveMode,
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

        printPerformanceSummary(metrics, processCount, total_time, !args.no_display, args.saveMode);

        DXRT_TRY_CATCH_END
        return 0;
    }

private:
    std::unique_ptr<FactoryT> factory_;
    std::string model_path_;

    CommandLineArgs parseCommandLine(int argc, char* argv[]) {
        CommandLineArgs args;
        std::string app_name = factory_->getModelName() + " OBB Post-Processing Sync Example";
        cxxopts::Options options(app_name, app_name + " application usage ");
        options.add_options()
            ("m, model_path", "OBB detection model file (.dxnn, required)",
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
            dxapp::fatal_error("[ERROR] Model path is required. Use -m or --model_path option.\n"
                "        -> Download:  ./setup.sh --models <model_name>\n"
                "        -> Or use:    ./run_demo.sh  (auto-downloads demo models)");
        }
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
        if (sourceCount != 1) {
            dxapp::fatal_error("[ERROR] Please specify exactly one input source.");
        }
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

    bool processSingleFrame(
        const cv::Mat& input_frame, cv::Mat& display_image,
        dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor,
        IPostprocessor<OBBResult>& postprocessor,
        IVisualizer<OBBResult>& visualizer,
        SyncProfilingMetrics& metrics,
        cv::VideoWriter& writer, bool no_display, bool saveMode, double t_read,
        int frameIdx = 0,
        const std::string& dumpTensorsDir = "",
        bool dumpPerFrameDir = false) {

        if (input_frame.empty()) return false;

        auto t0 = std::chrono::high_resolution_clock::now();
        dxapp::displayResize(input_frame, display_image, SHOW_WINDOW_SIZE_W, SHOW_WINDOW_SIZE_H);
        PreprocessContext ctx;
        cv::Mat preprocessed;
        preprocessor.process(display_image, preprocessed, ctx);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto outputs = ie.Run(preprocessed.data, nullptr, nullptr);
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

        std::vector<OBBResult> results;
        auto t_post_start = std::chrono::high_resolution_clock::now();
        try {
            results = postprocessor.process(outputs, ctx);
        } catch (const std::exception& e) {
            std::cerr << "[DXAPP] [ER] Postprocess error: " << e.what() << std::endl;
            // Auto-dump tensors on exception for debugging
            if (dumpTensorsDir.empty()) {
                static const std::string kErrorDumpDir = "error_tensors";
                dumpTensorsToFiles(kErrorDumpDir, preprocessed, outputs);
                if (verbose_) {
                    std::cout << "[INFO] Auto-dumped tensors on exception to: "
                              << fs::absolute(kErrorDumpDir).string() << std::endl;
                }
            }
            return false;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double t_postprocess = std::chrono::duration<double, std::milli>(t3 - t_post_start).count();

        // Print OBB detection results for pipeline parsing
        for (const auto& o : results) {
            std::string oname = o.class_name.empty() ? std::to_string(o.class_id) : o.class_name;
            for (auto& ch : oname) if (ch == ' ') ch = '_';
            if (verbose_) {
                std::cout << "[OBB] " << oname
                          << " " << std::fixed << std::setprecision(4) << o.confidence
                          << " " << std::fixed << std::setprecision(1) << (o.angle * 180.0 / 3.14159265358979)
                          << std::endl;
            }
        }


        // --- Numerical verification dump (DXAPP_VERIFY=1) ---
        verify::dumpVerifyJson(results, model_path_,
            "obb_detection", display_image.rows, display_image.cols);

        auto render_start = std::chrono::high_resolution_clock::now();
        cv::Mat result_frame = visualizer.draw(display_image, results, ctx);

        bool quit_requested = false;
        if (!result_frame.empty()) {
            if (saveMode) writer << result_frame;
            { const char* _sv=std::getenv("DXAPP_SAVE_IMAGE"); if(_sv&&*_sv)cv::imwrite(_sv,result_frame); }
            if (!no_display) {
                cv::imshow("Output", result_frame);
                if (dxapp::windowShouldClose("Output")) quit_requested = true;
            }
        }
        auto render_end = std::chrono::high_resolution_clock::now();
        double t_render = std::chrono::duration<double, std::milli>(render_end - render_start).count();

        metrics.sum_read += t_read;
        metrics.sum_preprocess += t_preprocess;
        metrics.sum_inference += t_inference;
        metrics.sum_postprocess += t_postprocess;
        metrics.sum_render += t_render;
        metrics.infer_completed++;

        return !quit_requested;
    }

    void processImageFrames(
        const std::vector<std::string>& imageFiles, int loopTest,
        cv::Mat& display_image, dxrt::InferenceEngine& ie,
        IPreprocessor& preprocessor, IPostprocessor<OBBResult>& postprocessor,
        IVisualizer<OBBResult>& visualizer, SyncProfilingMetrics& metrics,
        int& processCount, cv::VideoWriter& writer, bool no_display, bool saveMode,
        const std::string& runDir = "", bool dumpEnabled = false) {
        for (int i = 0; i < loopTest; ++i) {
            std::string currentImagePath = imageFiles[i % imageFiles.size()];
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
        IPreprocessor& preprocessor, IPostprocessor<OBBResult>& postprocessor,
        IVisualizer<OBBResult>& visualizer, SyncProfilingMetrics& metrics,
        int& processCount, cv::VideoWriter& writer, bool no_display, bool saveMode,
        int loopTest, const std::string& videoFile,
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

#endif  // OBB_RUNNER_HPP
