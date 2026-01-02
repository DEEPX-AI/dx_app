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
 * @brief Function to handle keyboard input in a separate thread
 */
void keyboard_input_handler() {
    std::string input;
    while (!g_should_exit.load()) {
        std::cout << "Press 'q' + Enter to quit: ";
        std::getline(std::cin, input);
        if (input == "q" || input == "Q") {
            g_should_exit.store(true);
            std::cout << "[INFO] Exit requested by user" << std::endl;
            break;
        }
    }
}

/**
 * @brief Video file processing using GStreamer
 * GStreamer provides excellent video file support with hardware acceleration options
 */
class GStreamerVideoPlayer {
   private:
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* decoder = nullptr;
    GstElement* converter = nullptr;
    GstElement* scaler = nullptr;
    GstElement* sink = nullptr;

    uint32_t output_width;
    uint32_t output_height;
    bool is_running = false;
    bool eos_received = false;

   public:
    GStreamerVideoPlayer(uint32_t width, uint32_t height)
        : output_width(width), output_height(height) {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
    }

    ~GStreamerVideoPlayer() { cleanup(); }

    bool initialize(const std::string& video_path) {
        // Create pipeline elements
        pipeline = gst_pipeline_new("video-pipeline");
        source = gst_element_factory_make("filesrc", "source");
        decoder = gst_element_factory_make("decodebin", "decoder");
        converter = gst_element_factory_make("videoconvert", "converter");
        scaler = gst_element_factory_make("videoscale", "scaler");
        sink = gst_element_factory_make("appsink", "sink");

        if (!pipeline || !source || !decoder || !converter || !scaler || !sink) {
            std::cerr << "[ERROR] Failed to create GStreamer elements" << std::endl;
            return false;
        }

        // Configure source
        g_object_set(G_OBJECT(source), "location", video_path.c_str(), NULL);

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
        gst_bin_add_many(GST_BIN(pipeline), source, decoder, converter, scaler, sink, NULL);

        // Link source to decoder
        if (!gst_element_link(source, decoder)) {
            std::cerr << "[ERROR] Failed to link source to decoder" << std::endl;
            return false;
        }

        // Link converter, scaler, and sink
        if (!gst_element_link_many(converter, scaler, sink, NULL)) {
            std::cerr << "[ERROR] Failed to link converter to sink" << std::endl;
            return false;
        }

        // Connect pad-added signal for dynamic linking (decoder has dynamic pads)
        g_signal_connect(decoder, "pad-added", G_CALLBACK(on_pad_added), converter);

        // Connect bus signals for EOS detection
        GstBus* bus = gst_element_get_bus(pipeline);
        gst_bus_add_signal_watch(bus);
        g_signal_connect(bus, "message", G_CALLBACK(on_bus_message), this);
        gst_object_unref(bus);

        std::cout << "[INFO] GStreamer video player initialized successfully" << std::endl;
        std::cout << "[INFO] Video file: " << video_path << std::endl;
        std::cout << "[INFO] Output: " << output_width << "x" << output_height << std::endl;

        return true;
    }

    static void on_pad_added(GstElement* element, GstPad* pad, gpointer data) {
        (void)element;
        GstElement* converter = GST_ELEMENT(data);
        GstPad* sink_pad = gst_element_get_static_pad(converter, "sink");

        if (gst_pad_is_linked(sink_pad)) {
            gst_object_unref(sink_pad);
            return;
        }

        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (caps) {
            GstStructure* structure = gst_caps_get_structure(caps, 0);
            const gchar* name = gst_structure_get_name(structure);

            if (g_str_has_prefix(name, "video/")) {
                gst_pad_link(pad, sink_pad);
                std::cout << "[INFO] Linked video pad: " << name << std::endl;
            }
            gst_caps_unref(caps);
        }
        gst_object_unref(sink_pad);
    }

    static gboolean on_bus_message(GstBus* bus, GstMessage* message, gpointer data) {
        (void)bus;
        GStreamerVideoPlayer* player = static_cast<GStreamerVideoPlayer*>(data);

        switch (GST_MESSAGE_TYPE(message)) {
            case GST_MESSAGE_EOS:
                std::cout << "[INFO] End of stream reached" << std::endl;
                player->eos_received = true;
                g_should_exit.store(true);
                break;
            case GST_MESSAGE_ERROR: {
                GError* error;
                gchar* debug;
                gst_message_parse_error(message, &error, &debug);
                std::cerr << "[ERROR] GStreamer error: " << error->message << std::endl;
                if (debug) {
                    std::cerr << "[DEBUG] " << debug << std::endl;
                }
                g_error_free(error);
                g_free(debug);
                g_should_exit.store(true);
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError* error;
                gchar* debug;
                gst_message_parse_warning(message, &error, &debug);
                std::cout << "[WARNING] GStreamer warning: " << error->message << std::endl;
                if (debug) {
                    std::cout << "[DEBUG] " << debug << std::endl;
                }
                g_error_free(error);
                g_free(debug);
                break;
            }
            default:
                break;
        }
        return TRUE;
    }

    bool start() {
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[ERROR] Failed to start GStreamer pipeline" << std::endl;
            return false;
        }

        is_running = true;
        std::cout << "[INFO] GStreamer video pipeline started" << std::endl;
        return true;
    }

    bool getFrame(uint8_t* output_buffer) {
        if (!is_running || eos_received) return false;

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) {
            if (!eos_received) {
                // Check if we've reached end of stream
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;

        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            size_t expected_size = output_width * output_height * 3;
            if (map.size >= expected_size) {
                memcpy(output_buffer, map.data, expected_size);
            } else {
                std::cerr << "[WARNING] Buffer size mismatch: got " << map.size << ", expected "
                          << expected_size << std::endl;
                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);
                return false;
            }
            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
        return true;
    }

    bool getFrameAsMat(cv::Mat& output_mat) {
        if (!is_running || eos_received) return false;

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) {
            if (!eos_received) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;

        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            size_t expected_size = output_width * output_height * 3;
            if (map.size >= expected_size) {
                // Create OpenCV Mat from GStreamer buffer data
                cv::Mat temp(output_height, output_width, CV_8UC3, map.data);
                temp.copyTo(output_mat);
            } else {
                std::cerr << "[WARNING] Buffer size mismatch: got " << map.size << ", expected "
                          << expected_size << std::endl;
                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);
                return false;
            }
            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
        return true;
    }

    bool isEOS() const { return eos_received; }

    void stop() {
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        is_running = false;
    }

    void cleanup() {
        stop();
        if (pipeline) {
            GstBus* bus = gst_element_get_bus(pipeline);
            if (bus) {
                gst_bus_remove_signal_watch(bus);
                gst_object_unref(bus);
            }
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
    }
};

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "";
    std::string videoPath = "";
    int processCount = 0;
    uint32_t input_w = 640, input_h = 480;

    std::queue<uint8_t> keyQueue;

    std::string app_name = "GStreamer Video test";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "v, video_path", "Video file path (required)", cxxopts::value<std::string>(videoPath))(
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

    if (videoPath.empty()) {
        std::cerr << "[ERROR] Video path is required." << std::endl;
        exit(1);
    }

    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(modelPath, io);

    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    GStreamerVideoPlayer player(input_w, input_h);

    if (!player.initialize(videoPath)) {
        std::cerr << "[ERROR] Failed to initialize GStreamer video player" << std::endl;
        return -1;
    }

    if (!player.start()) {
        std::cerr << "[ERROR] Failed to start GStreamer pipeline" << std::endl;
        return -1;
    }

    std::vector<std::vector<uint8_t>> inputTensors(10);
    for (auto& inputTensor : inputTensors) {
        inputTensor = std::vector<uint8_t>(ie.GetInputSize());
    }

    // Start keyboard input handler thread
    std::thread keyboard_thread(keyboard_input_handler);

    int index = 0;
    auto s = std::chrono::high_resolution_clock::now();
    cv::Mat display_frame;
    std::cout << "[INFO] Starting video processing. Press 'q' + Enter to quit or ESC key in "
                 "display window."
              << std::endl;

    while (!g_should_exit.load() && !player.isEOS()) {
        // Get frame for display
        if (!player.getFrameAsMat(display_frame)) {
            if (player.isEOS()) {
                std::cout << "[INFO] End of video file reached." << std::endl;
                break;
            }
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
        cv::imshow("GStreamer Video Player", display_frame);

        // Check for ESC key press (non-blocking)
        auto key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') {  // ESC key or 'q'
            std::cout << "[INFO] Exit requested via display window" << std::endl;
            g_should_exit.store(true);
            break;
        }

        // Optional progress indicator
        if (processCount % 100 == 0) {
            std::cout << "[INFO] Processed " << processCount << " frames" << std::endl;
        }
    }

    // Clean up
    std::cout << "[INFO] Stopping video processing..." << std::endl;
    player.stop();
    cv::destroyAllWindows();

    // Wait for keyboard thread to finish
    g_should_exit.store(true);
    if (keyboard_thread.joinable()) {
        keyboard_thread.join();
    }

    auto e = std::chrono::high_resolution_clock::now();
    double fps_actual =
        processCount /
        (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0);
    std::cout << "[INFO] Processed " << processCount << " frames total" << std::endl;
    std::cout << "[INFO] Processing time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms"
              << std::endl;
    std::cout << "[INFO] Average FPS: " << fps_actual << std::endl;

    DXRT_TRY_CATCH_END
    return 0;
}