#include <dxrt/dxrt_api.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Global flag for exit condition
std::atomic<bool> g_should_exit(false);

/**
 * @brief RTSP stream processing using GStreamer
 * GStreamer provides excellent RTSP support with hardware acceleration options
 */
class GStreamerRTSPReceiver {
   private:
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* decoder = nullptr;
    GstElement* converter = nullptr;
    GstElement* sink = nullptr;

    uint32_t output_width;
    uint32_t output_height;
    bool is_running = false;

   public:
    GStreamerRTSPReceiver(uint32_t width, uint32_t height)
        : output_width(width), output_height(height) {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
    }

    ~GStreamerRTSPReceiver() { cleanup(); }

    bool initialize(const std::string& rtsp_url) {
        // Create pipeline elements
        pipeline = gst_pipeline_new("rtsp-pipeline");
        source = gst_element_factory_make("rtspsrc", "source");
        decoder = gst_element_factory_make("avdec_h264", "decoder");
        converter = gst_element_factory_make("videoconvert", "converter");
        sink = gst_element_factory_make("appsink", "sink");

        if (!pipeline || !source || !decoder || !converter || !sink) {
            std::cerr << "[ERROR] Failed to create GStreamer elements" << std::endl;
            return false;
        }

        // Configure source
        g_object_set(G_OBJECT(source), "location", rtsp_url.c_str(), NULL);
        g_object_set(G_OBJECT(source), "latency", 0, NULL);
        g_object_set(G_OBJECT(source), "drop-on-latency", TRUE, NULL);

        // Configure sink
        GstCaps* caps =
            gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", "width", G_TYPE_INT,
                                output_width, "height", G_TYPE_INT, output_height, NULL);
        g_object_set(G_OBJECT(sink), "caps", caps, NULL);
        g_object_set(G_OBJECT(sink), "emit-signals", TRUE, NULL);
        g_object_set(G_OBJECT(sink), "max-buffers", 1, NULL);
        g_object_set(G_OBJECT(sink), "drop", TRUE, NULL);
        gst_caps_unref(caps);

        // Add elements to pipeline
        gst_bin_add_many(GST_BIN(pipeline), source, decoder, converter, sink, NULL);

        // Link elements (note: rtspsrc uses dynamic pads)
        if (!gst_element_link_many(decoder, converter, sink, NULL)) {
            std::cerr << "[ERROR] Failed to link GStreamer elements" << std::endl;
            return false;
        }

        // Connect pad-added signal for dynamic linking
        g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), decoder);

        std::cout << "[INFO] GStreamer RTSP receiver initialized successfully" << std::endl;
        return true;
    }

    static void on_pad_added(GstElement* element, GstPad* pad, gpointer data) {
        (void)element;
        GstElement* decoder = GST_ELEMENT(data);
        GstPad* sink_pad = gst_element_get_static_pad(decoder, "sink");

        if (gst_pad_is_linked(sink_pad)) {
            gst_object_unref(sink_pad);
            return;
        }

        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (caps) {
            GstStructure* structure = gst_caps_get_structure(caps, 0);
            const gchar* name = gst_structure_get_name(structure);

            if (g_str_has_prefix(name, "video/x-h264")) {
                gst_pad_link(pad, sink_pad);
                std::cout << "[INFO] Linked video pad" << std::endl;
            }
            gst_caps_unref(caps);
        }
        gst_object_unref(sink_pad);
    }

    bool start() {
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[ERROR] Failed to start GStreamer pipeline" << std::endl;
            return false;
        }

        is_running = true;
        std::cout << "[INFO] GStreamer pipeline started" << std::endl;
        return true;
    }

    bool getFrame(uint8_t* output_buffer) {
        if (!is_running) return false;

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) {
            return false;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;

        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            memcpy(output_buffer, map.data, output_width * output_height * 3);
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
        GstMapInfo map;

        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // Create OpenCV Mat from GStreamer buffer data
            cv::Mat temp(output_height, output_width, CV_8UC3, map.data);
            temp.copyTo(output_mat);
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
    std::string rtspURL = "";
    int processCount = 0;
    uint32_t input_w = 640, input_h = 480;

    std::queue<uint8_t> keyQueue;

    std::string app_name = "GStreamer RTSP test";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "r, rtsp_url", "RTSP stream URL", cxxopts::value<std::string>(rtspURL))(
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

    if (!rtspURL.empty()) {
        GStreamerRTSPReceiver receiver(input_w, input_h);

        if (!receiver.initialize(rtspURL)) {
            std::cerr << "[ERROR] Failed to initialize GStreamer RTSP receiver" << std::endl;
            return -1;
        }

        if (!receiver.start()) {
            std::cerr << "[ERROR] Failed to start GStreamer pipeline" << std::endl;
            return -1;
        }

        std::vector<std::vector<uint8_t>> inputTensors(10);
        for (auto& inputTensor : inputTensors) {
            inputTensor = std::vector<uint8_t>(ie.GetInputSize());
        }

        int index = 0;
        cv::Mat display_frame;
        std::cout << "[INFO] Starting RTSP processing. Press ESC key in display window."
                  << std::endl;

        while (!g_should_exit.load()) {
            // Get frame for display
            if (!receiver.getFrameAsMat(display_frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Copy frame data for inference
            cv::Mat inference_frame;
            cv::resize(display_frame, inference_frame, cv::Size(input_w, input_h));
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
            cv::imshow("GStreamer RTSP Feed", display_frame);

            // Check for ESC key press (non-blocking)
            auto key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') {  // ESC key or 'q'
                std::cout << "[INFO] Exit requested via display window" << std::endl;
                g_should_exit.store(true);
                break;
            }
        }

        // Clean up
        std::cout << "[INFO] Stopping RTSP processing..." << std::endl;
        receiver.stop();
        cv::destroyAllWindows();

        std::cout << "[INFO] Processed " << processCount << " frames total" << std::endl;
    }

    DXRT_TRY_CATCH_END
    return 0;
}