#include <dxrt/dxrt_api.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Global flag for exit condition
std::atomic<bool> g_should_exit(false);

/**
 * @brief Camera capture using GStreamer
 * GStreamer provides excellent camera support with hardware acceleration options
 */
class GStreamerCameraCapture {
   private:
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* converter = nullptr;
    GstElement* sink = nullptr;

    uint32_t camera_width = 0;  // Original camera resolution
    uint32_t camera_height = 0;
    bool is_running = false;

   public:
    GStreamerCameraCapture() {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
    }

    ~GStreamerCameraCapture() { cleanup(); }

    bool initialize(const std::string& device, uint32_t fps = 30) {
        // Create pipeline elements
        pipeline = gst_pipeline_new("camera-pipeline");
        source = gst_element_factory_make("v4l2src", "source");
        converter = gst_element_factory_make("videoconvert", "converter");
        sink = gst_element_factory_make("appsink", "sink");

        if (!pipeline || !source || !converter || !sink) {
            std::cerr << "[ERROR] Failed to create GStreamer elements" << std::endl;
            return false;
        }

        // Configure source
        g_object_set(G_OBJECT(source), "device", device.c_str(), NULL);

        // Configure caps for source (let camera use its native resolution and format)
        GstCaps* src_caps =
            gst_caps_new_simple("video/x-raw", "framerate", GST_TYPE_FRACTION, fps, 1, NULL);

        // Configure sink to convert to BGR format
        GstCaps* sink_caps =
            gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR", NULL);
        g_object_set(G_OBJECT(sink), "caps", sink_caps, NULL);
        g_object_set(G_OBJECT(sink), "emit-signals", TRUE, NULL);
        g_object_set(G_OBJECT(sink), "max-buffers", 1, NULL);
        g_object_set(G_OBJECT(sink), "drop", TRUE, NULL);

        // Add elements to pipeline
        gst_bin_add_many(GST_BIN(pipeline), source, converter, sink, NULL);

        // Link elements with caps
        if (!gst_element_link_filtered(source, converter, src_caps)) {
            std::cerr << "[ERROR] Failed to link source to converter" << std::endl;
            gst_caps_unref(src_caps);
            return false;
        }

        if (!gst_element_link_filtered(converter, sink, sink_caps)) {
            std::cerr << "[ERROR] Failed to link converter to sink" << std::endl;
            gst_caps_unref(src_caps);
            gst_caps_unref(sink_caps);
            return false;
        }

        gst_caps_unref(src_caps);
        gst_caps_unref(sink_caps);

        std::cout << "[INFO] GStreamer camera capture initialized successfully" << std::endl;
        std::cout << "[INFO] Device: " << device << " @ " << fps << " FPS" << std::endl;
        std::cout << "[INFO] Will detect actual camera resolution at runtime" << std::endl;
        std::cout << "[INFO] videoconvert will handle YUV2/YUYV -> BGR conversion automatically"
                  << std::endl;

        return true;
    }

    bool start() {
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[ERROR] Failed to start GStreamer pipeline" << std::endl;
            return false;
        }

        is_running = true;
        std::cout << "[INFO] GStreamer camera pipeline started" << std::endl;
        return true;
    }

    bool getFrame(uint8_t* output_buffer) {
        if (!is_running) return false;

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) {
            return false;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);

        // Get actual frame dimensions from caps
        GstStructure* structure = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(structure, "width", (int*)&camera_width);
        gst_structure_get_int(structure, "height", (int*)&camera_height);

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // Calculate buffer size based on actual format
            size_t buffer_size = camera_width * camera_height * 3;  // Assume 3 channels for now
            memcpy(output_buffer, map.data, std::min(buffer_size, (size_t)map.size));
            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
        return true;
    }

    bool getFrameAsMat(cv::Mat& output_mat) {
        if (!is_running) return false;

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) {
            return false;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);

        // Get actual frame dimensions from caps
        GstStructure* structure = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(structure, "width", (int*)&camera_width);
        gst_structure_get_int(structure, "height", (int*)&camera_height);

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // Get format information
            const gchar* format_str = gst_structure_get_string(structure, "format");

            // Since we configured sink to output BGR, we should get BGR format
            // videoconvert will handle YUV2/YUYV -> BGR conversion automatically
            cv::Mat temp(camera_height, camera_width, CV_8UC3, map.data);

            if (g_strcmp0(format_str, "BGR") == 0) {
                // Direct BGR format - just copy
                temp.copyTo(output_mat);
            } else if (g_strcmp0(format_str, "RGB") == 0) {
                // Convert RGB to BGR
                cv::cvtColor(temp, output_mat, cv::COLOR_RGB2BGR);
            } else {
                // Fallback: assume BGR format (should not happen with our caps)
                std::cout << "[WARNING] Unexpected format: " << format_str << ", assuming BGR"
                          << std::endl;
                temp.copyTo(output_mat);
            }

            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
        return true;
    }

    void stop() {
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        is_running = false;
    }

    uint32_t getCameraWidth() const { return camera_width; }

    uint32_t getCameraHeight() const { return camera_height; }

    void cleanup() {
        stop();
        if (pipeline) {
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
    }
};

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "";
    std::string device = "/dev/video0";
    int processCount = 0;
    uint32_t input_w = 640, input_h = 480;
    uint32_t fps = 30;

    std::queue<uint8_t> keyQueue;

    std::string app_name = "GStreamer Camera test";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "d, device", "Camera device path (required)",
        cxxopts::value<std::string>(device)->default_value("/dev/video0"))(
        "fps", "Camera FPS", cxxopts::value<uint32_t>(fps)->default_value("30"))(
        "width, input_width", "input width size",
        cxxopts::value<uint32_t>(input_w)->default_value("640"))(
        "height, input_height", "input height size",
        cxxopts::value<uint32_t>(input_h)->default_value("640"))("h, help", "print usage");
    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path is required." << std::endl;
        exit(1);
    }

    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(modelPath, io);

    if (device.empty() == false) {
        GStreamerCameraCapture capture;

        if (!capture.initialize(device, fps)) {
            std::cerr << "[ERROR] Failed to initialize GStreamer camera capture" << std::endl;
            return -1;
        }

        if (!capture.start()) {
            std::cerr << "[ERROR] Failed to start GStreamer pipeline" << std::endl;
            return -1;
        }

        std::vector<std::vector<uint8_t>> inputTensors(10);
        for (auto& inputTensor : inputTensors) {
            inputTensor = std::vector<uint8_t>(ie.GetInputSize());
        }

        int index = 0;
        auto s = std::chrono::high_resolution_clock::now();
        cv::Mat display_frame;
        std::cout
            << "[INFO] Starting camera processing. Press 'q' to quit or ESC key in display window."
            << std::endl;

        bool camera_info_printed = false;
        while (!g_should_exit.load()) {
            // Get frame for display
            if (!capture.getFrameAsMat(display_frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Print camera resolution once
            if (!camera_info_printed && capture.getCameraWidth() > 0) {
                std::cout << "[INFO] Detected camera resolution: " << capture.getCameraWidth()
                          << "x" << capture.getCameraHeight() << std::endl;
                std::cout << "[INFO] Inference resolution: " << input_w << "x" << input_h
                          << std::endl;
                std::cout << "[INFO] Display frame size: " << display_frame.cols << "x"
                          << display_frame.rows << " channels: " << display_frame.channels()
                          << std::endl;
                camera_info_printed = true;
            }

            // Process frame for inference: resize and ensure RGB format
            cv::Mat inference_frame;
            cv::resize(display_frame, inference_frame, cv::Size(input_w, input_h));
            cv::cvtColor(inference_frame, inference_frame, cv::COLOR_BGR2RGB);

            // Convert to RGB format for inference if needed
            // Copy to inference tensor
            memcpy(inputTensors[index].data(), inference_frame.data, input_w * input_h * 3);

            keyQueue.push(ie.RunAsync(inputTensors[index].data()));
            index = (index + 1) % inputTensors.size();
            processCount++;

            // Process inference results
            if (!keyQueue.empty()) {
                auto outputs = ie.Wait(keyQueue.front());
                keyQueue.pop();
                std::cout << "[DXAPP] [INFO] post processing result: " << outputs.size() << " items"
                          << std::endl;
                // Optional: Draw inference results on display frame
                // You can add result visualization here
            }

            // Display the frame
            cv::imshow("GStreamer Camera Feed", display_frame);

            // Check for ESC key press (non-blocking)
            auto key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') {  // ESC key or 'q'
                std::cout << "[INFO] Exit requested via display window" << std::endl;
                g_should_exit.store(true);
                break;
            }
        }

        // Clean up
        std::cout << "[INFO] Stopping camera processing..." << std::endl;
        capture.stop();
        cv::destroyAllWindows();

        auto e = std::chrono::high_resolution_clock::now();
        double fps_actual =
            processCount /
            (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0);
        std::cout << "[INFO] Processed " << processCount << " frames" << std::endl;
        std::cout << "[INFO] Actual FPS: " << fps_actual << std::endl;
    } else {
        std::cout << "[INFO] Running without camera input" << std::endl;
    }

    DXRT_TRY_CATCH_END
    return 0;
}