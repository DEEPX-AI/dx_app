#include <dxrt/dxrt_api.h>

#include <chrono>  // For timing measurements
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iomanip>  // For std::setprecision
#include <iostream>
#include <memory>  // For smart pointers
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>  // STL vector container

#include "object_detection_util.h"

/**
 * @brief Multi-channel asynchronous post-processing example for YOLOv5s object detection model.
 *
 * - Supports multiple independent channels with different input sources
 * - Each channel runs in its own thread performing: frame acquisition, preprocessing, inference,
 * postprocessing, and display
 * - Supports image, video, camera, and RTSP stream input sources
 * - Independent visualization and result saving for each channel
 * - Command-line options allow configuration of model path, input sources, loop count, and display
 * options
 */

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN

    // // Initialize X11 threads for multi-threaded GUI operations
    // // This must be called before any OpenCV GUI operations in multi-threaded environment
    // cv::startWindowThread();

    // // Additional X11 thread safety initialization
    // setenv("XInitThreads", "1", 1);

    std::string modelPath = "", inputList = "";
    bool fps_only = false, save_video = false;
    int loopTest = 1;
    float conf_threshold = 0.25f, nms_threshold = 0.3f, iou_threshold = 0.45f;

    std::string app_name = "YOLOv5s Multi-Channel Independent Processing Example";
    cxxopts::Options options(app_name, app_name + " application usage");

    options.add_options()("m,model_path", "object detection model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "vlist",
        "input file path list (ex. --vlist sample.jpg,sample.mp4,/dev/video0 ..., required)",
        cxxopts::value<std::string>(inputList))(
        "l,loop", "Number of inference iterations to run for each channel",
        cxxopts::value<int>(loopTest)->default_value("1"))(
        "no-display", "will not visualize, only show fps",
        cxxopts::value<bool>(fps_only)->default_value("false"))(
        "save-video", "save result video for each channel",
        cxxopts::value<bool>(save_video)->default_value("false"))(
        "conf-threshold", "confidence threshold for detection",
        cxxopts::value<float>(conf_threshold)->default_value("0.25"))(
        "nms-threshold", "NMS threshold",
        cxxopts::value<float>(nms_threshold)->default_value("0.3"))(
        "iou-threshold", "IOU threshold",
        cxxopts::value<float>(iou_threshold)->default_value("0.45"))("h,help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // Validate required arguments
    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path is required. Use -m or --model_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    if (inputList.empty()) {
        std::cerr << "[ERROR] Input list is required. Use --vlist option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    // Parse input sources
    std::vector<std::string> inputPaths = dxapp::common::split(inputList, ',');
    if (inputPaths.empty()) {
        std::cerr << "[ERROR] No valid input paths found" << std::endl;
        return -1;
    }

    std::cout << "=== " << app_name << " ===" << std::endl;
    std::cout << "[INFO] Model: " << modelPath << std::endl;
    std::cout << "[INFO] Number of channels: " << inputPaths.size() << std::endl;
    std::cout << "[INFO] Loop count per channel: " << loopTest << std::endl;
    std::cout << "[INFO] Confidence threshold: " << conf_threshold << std::endl;
    std::cout << "[INFO] NMS threshold: " << nms_threshold << std::endl;
    std::cout << "[INFO] IOU threshold: " << iou_threshold << std::endl;

    // Create multi-channel processor
    MultiChannelProcessor processor;

    // Add each input source as a separate channel
    for (size_t i = 0; i < inputPaths.size(); ++i) {
        ChannelProcessor::ChannelConfig config;
        config.input_source = inputPaths[i];
        config.channel_id = static_cast<int>(i);
        config.model_path = modelPath;
        config.loop_count = loopTest;
        config.target_fps = 0.0;
        config.no_display = fps_only;
        config.save_video = save_video;
        config.conf_threshold = conf_threshold;
        config.nms_threshold = nms_threshold;
        config.iou_threshold = iou_threshold;
        config.display_width = 960;
        config.display_height = 640;

        if (save_video) {
            config.output_video_path = "channel_" + std::to_string(i) + "_result.mp4";
        }

        processor.addChannel(config);

        std::cout << "[INFO] Added channel " << i << ": " << inputPaths[i] << std::endl;
    }

    // Initialize all channels
    std::cout << "\n[INFO] Initializing all channels..." << std::endl;
    if (!processor.initializeAll()) {
        std::cerr << "[ERROR] Failed to initialize some channels" << std::endl;
        return -1;
    }

    // Start all channels
    std::cout << "[INFO] Starting all channels..." << std::endl;
    if (!processor.startAll()) {
        std::cerr << "[ERROR] Failed to start some channels" << std::endl;
        return -1;
    }

    // Monitor progress
    std::cout << "[INFO] All channels started. Processing..." << std::endl;
    if (!fps_only) {
        std::cout << "[INFO] Press 'q' in any window to stop that channel" << std::endl;
        std::cout << "[INFO] Close this terminal or press Ctrl+C to stop all channels" << std::endl;
    }

    // Wait for all channels to complete
    auto start_time = std::chrono::high_resolution_clock::now();

    // Periodic status update
    std::thread status_thread([&processor, &start_time]() {
        while (processor.isAnyRunning()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(current_time - start_time).count();

            std::cout << "[STATUS] Elapsed time: " << std::fixed << std::setprecision(1) << elapsed
                      << " seconds" << std::endl;
        }
    });

    // Wait for completion
    processor.joinAll();

    // Stop status thread
    if (status_thread.joinable()) {
        status_thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n[INFO] All channels completed!" << std::endl;
    std::cout << "[INFO] Total processing time: " << std::fixed << std::setprecision(2)
              << total_time << " seconds" << std::endl;

    // Print performance reports
    processor.printOverallReport();

    // Clean up OpenCV windows
    std::cout << "\n[SUCCESS] Multi-channel processing completed successfully!" << std::endl;

    DXRT_TRY_CATCH_END
    return 0;
}
