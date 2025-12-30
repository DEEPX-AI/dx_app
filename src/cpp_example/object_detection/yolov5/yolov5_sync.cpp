// A structured, template-style C++ example for YOLOv5 synchronous inference.
//
// This example is intentionally aligned with `yolov5_sync.py`:
//   - YOLOv5 class encapsulating model, preprocess, postprocess, and drawing
//   - image_inference / stream_inference entrypoints
//   - simple CLI that selects exactly one of: image, video, camera, rtsp

#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolov5_postprocess.h"

namespace {

struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    double sum_postprocess = 0.0;
    double sum_render = 0.0;
};

cv::Scalar get_class_color(int class_id) {
    // Deterministic color per class id (simple hash, no global RNG state).
    unsigned seed = static_cast<unsigned>(class_id * 123457u + 98765u);
    unsigned b = (seed * 16807u + 3u) & 0xFFu;
    unsigned g = (seed * 48271u + 7u) & 0xFFu;
    unsigned r = (seed * 69621u + 11u) & 0xFFu;
    return cv::Scalar(static_cast<double>(b), static_cast<double>(g), static_cast<double>(r));
}

void print_sync_performance_summary(const ProfilingMetrics& m, int total_frames,
                                    double total_time_sec, bool display) {
    if (total_frames <= 0) {
        std::cout << "[WARNING] No frames processed" << std::endl;
        return;
    }

    auto avg = [&](double v) { return v / static_cast<double>(total_frames); };

    double avg_read = avg(m.sum_read);
    double avg_pre = avg(m.sum_preprocess);
    double avg_inf = avg(m.sum_inference);
    double avg_post = avg(m.sum_postprocess);

    auto fps = [](double ms) { return ms > 0.0 ? 1000.0 / ms : 0.0; };

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Read" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_read << " ms     " << std::setw(6)
              << std::setprecision(1) << fps(avg_read) << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Preprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_pre << " ms     " << std::setw(6)
              << std::setprecision(1) << fps(avg_pre) << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Inference" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_inf << " ms     " << std::setw(6)
              << std::setprecision(1) << fps(avg_inf) << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Postprocess" << std::right
              << std::setw(8) << std::fixed << std::setprecision(2) << avg_post << " ms     "
              << std::setw(6) << std::setprecision(1) << fps(avg_post) << " FPS" << std::endl;

    if (display) {
        double avg_render = avg(m.sum_render);
        std::cout << " " << std::left << std::setw(15) << "Display" << std::right
                  << std::setw(8) << std::fixed << std::setprecision(2) << avg_render
                  << " ms     " << std::setw(6) << std::setprecision(1) << fps(avg_render)
                  << " FPS" << std::endl;
    }

    std::cout << "--------------------------------------------------" << std::endl;
    double overall_fps = total_time_sec > 0.0 ? total_frames / total_time_sec : 0.0;
    std::cout << " " << std::left << std::setw(19) << "Total Frames"
              << " :    " << total_frames << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Time"
              << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
              << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Overall FPS"
              << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
              << std::endl;
    std::cout << "==================================================" << std::endl;
}

class YOLOv5 {
public:
        explicit YOLOv5(const std::string& model_path)
                : model_path_(model_path),
                    obj_threshold_(0.25f),
                    score_threshold_(0.3f),
                    nms_threshold_(0.45f) {
        dxrt::InferenceOption option;
        ie_ = std::make_unique<dxrt::InferenceEngine>(model_path_, option);

        if (!dxapp::common::minversionforRTandCompiler(ie_.get())) {
            std::cerr << "[ERROR] The compiled model version is not compatible with the runtime. "
                         "Please recompile the model."
                      << std::endl;
            std::exit(1);
        }

        auto input_shape = ie_->GetInputs().front().shape();
        input_height_ = static_cast<int>(input_shape[1]);
        input_width_ = static_cast<int>(input_shape[2]);

        postprocess_ = std::make_unique<YOLOv5PostProcess>(
            input_width_, input_height_, obj_threshold_, score_threshold_, nms_threshold_,
            ie_->IsOrtConfigured());

        std::cout << "\n[INFO] Model loaded: " << model_path_ << std::endl;
        std::cout << "[INFO] Model input size (WxH): " << input_width_ << "x" << input_height_
                  << std::endl;
    }

    void image_inference(const std::string& image_path, bool display) {
        auto t_start = std::chrono::high_resolution_clock::now();

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Failed to load image: " << image_path << std::endl;
            std::exit(1);
        }

        std::cout << "\n[INFO] Input image: " << image_path << std::endl;
        std::cout << "[INFO] Image resolution (WxH): " << image.cols << "x" << image.rows
                  << std::endl;

        ProfilingMetrics metrics;

        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat pre = preprocess(image);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto outputs = ie_->Run(pre.data);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto detections = postprocess_->postprocess(outputs);
        auto t3 = std::chrono::high_resolution_clock::now();

        auto result = draw_detections(image, detections);
        auto t4 = std::chrono::high_resolution_clock::now();

        metrics.sum_preprocess =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        metrics.sum_inference =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        metrics.sum_postprocess =
            std::chrono::duration<double, std::milli>(t3 - t2).count();
        metrics.sum_render =
            std::chrono::duration<double, std::milli>(t4 - t3).count();

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() /
            1000.0;

        print_sync_performance_summary(metrics, 1, total_time, display);

        if (display) {
            cv::imshow("Output", result);
            cv::waitKey(0);
            cv::destroyAllWindows();
        } else {
            // For consistency with Python example, we simply save to current directory.
            cv::imwrite("result.jpg", result);
            std::cout << "[SUCCESS] Output saved: result.jpg" << std::endl;
        }
    }

    void stream_inference(const std::string& source, bool display) {
        cv::VideoCapture cap;

        const bool is_rtsp = source.rfind("rtsp://", 0) == 0;
        const bool is_camera = !is_rtsp &&
                               std::all_of(source.begin(), source.end(),
                                           [](unsigned char ch) { return std::isdigit(ch); });

        if (is_camera) {
            const int cam_index = std::stoi(source);
            cap.open(cam_index);
        } else {
            cap.open(source);
        }

        if (!cap.isOpened()) {
            std::cerr << "[ERROR] Failed to open input source: " << source << std::endl;
            std::exit(1);
        }

        const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (is_camera) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            std::cout << "\n[INFO] Camera index: " << source << std::endl;
        } else if (is_rtsp) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            std::cout << "\n[INFO] RTSP URL: " << source << std::endl;
        } else {
            std::cout << "\n[INFO] Video file: " << source << std::endl;
        }

        std::cout << "[INFO] Input source resolution (WxH): " << width << "x" << height
                  << std::endl;
        if (total_frames > 0) {
            std::cout << "[INFO] Total frames: " << total_frames << std::endl;
        }
        if (fps > 0.0) {
            std::cout << "[INFO] Input source FPS: " << std::fixed << std::setprecision(2) << fps
                      << std::endl;
        }

        std::cout << "\n[INFO] Starting inference..." << std::endl;

        ProfilingMetrics metrics;
        int frame_count = 0;

        auto t_stream_start = std::chrono::high_resolution_clock::now();

        while (true) {
            auto t_read_start = std::chrono::high_resolution_clock::now();

            cv::Mat frame_bgr;
            if (!cap.read(frame_bgr) || frame_bgr.empty()) {
                break;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            cv::Mat pre = preprocess(frame_bgr);
            auto t1 = std::chrono::high_resolution_clock::now();

            auto outputs = ie_->Run(pre.data);
            auto t2 = std::chrono::high_resolution_clock::now();

            auto detections = postprocess_->postprocess(outputs);
            auto t3 = std::chrono::high_resolution_clock::now();

            frame_count++;

            metrics.sum_read += std::chrono::duration<double, std::milli>(t0 - t_read_start).
                                  count();
            metrics.sum_preprocess +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            metrics.sum_inference +=
                std::chrono::duration<double, std::milli>(t2 - t1).count();
            metrics.sum_postprocess +=
                std::chrono::duration<double, std::milli>(t3 - t2).count();

            if (display) {
                auto result = draw_detections(frame_bgr, detections);
                cv::imshow("Output", result);
                int key = cv::waitKey(1) & 0xFF;

                auto t4 = std::chrono::high_resolution_clock::now();
                metrics.sum_render +=
                    std::chrono::duration<double, std::milli>(t4 - t3).count();

                if (key == 'q' || key == 27) {  // ESC
                    std::cout << "\n[INFO] User requested to quit" << std::endl;
                    break;
                }
            }
        }

        auto t_stream_end = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_stream_end - t_stream_start)
                .count() /
            1000.0;

        cap.release();
        cv::destroyAllWindows();

        print_sync_performance_summary(metrics, frame_count, total_time, display);
    }

private:
    cv::Mat preprocess(const cv::Mat& img_bgr) {
    img_height_ = img_bgr.rows;
    img_width_ = img_bgr.cols;

    gain_ = std::min(static_cast<float>(input_height_) / static_cast<float>(img_height_),
             static_cast<float>(input_width_) / static_cast<float>(img_width_));

    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    return letterbox(img_rgb);
    }

    cv::Mat letterbox(const cv::Mat& img_rgb) {
        // Based on Python implementation: resize with unchanged aspect ratio and pad.
        float r = gain_;

        int new_unpad_w = static_cast<int>(std::round(img_width_ * r));
        int new_unpad_h = static_cast<int>(std::round(img_height_ * r));

        cv::Mat resized;
        if (img_rgb.cols != new_unpad_w || img_rgb.rows != new_unpad_h) {
            cv::resize(img_rgb, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0,
                       cv::INTER_LINEAR);
        } else {
            resized = img_rgb;
        }

        float dw = (static_cast<float>(input_width_) - static_cast<float>(new_unpad_w)) / 2.0f;
        float dh = (static_cast<float>(input_height_) - static_cast<float>(new_unpad_h)) / 2.0f;

        int top = static_cast<int>(std::round(dh - 0.1f));
        int bottom = static_cast<int>(std::round(dh + 0.1f));
        int left = static_cast<int>(std::round(dw - 0.1f));
        int right = static_cast<int>(std::round(dw + 0.1f));

        pad_top_ = top;
        pad_left_ = left;

        cv::Mat out;
        cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT,
                           cv::Scalar(114, 114, 114));
        return out;
    }

    void scale_coordinates(YOLOv5Result& det) const {
        // reverse of letterbox + scaling, clamped to original image size
        auto clamp = [](float v, float lo, float hi) {
            return std::max(lo, std::min(v, hi));
        };

        det.box[0] =
            clamp((det.box[0] - static_cast<float>(pad_left_)) / gain_, 0.0f,
                  static_cast<float>(img_width_ - 1));
        det.box[1] =
            clamp((det.box[1] - static_cast<float>(pad_top_)) / gain_, 0.0f,
                  static_cast<float>(img_height_ - 1));
        det.box[2] =
            clamp((det.box[2] - static_cast<float>(pad_left_)) / gain_, 0.0f,
                  static_cast<float>(img_width_ - 1));
        det.box[3] =
            clamp((det.box[3] - static_cast<float>(pad_top_)) / gain_, 0.0f,
                  static_cast<float>(img_height_ - 1));
    }

    cv::Mat draw_detections(const cv::Mat& frame, std::vector<YOLOv5Result>& detections) const {
        cv::Mat result = frame.clone();

        for (auto& det : detections) {
            scale_coordinates(det);

            cv::Scalar color = get_class_color(det.class_id);

            cv::Point tl(static_cast<int>(det.box[0]), static_cast<int>(det.box[1]));
            cv::Point br(static_cast<int>(det.box[2]), static_cast<int>(det.box[3]));
            cv::rectangle(result, tl, br, color, 2);

            std::ostringstream oss;
            oss << det.class_name << ": " << std::fixed << std::setprecision(2)
                << det.confidence;
            std::string label = oss.str();

            int base_line = 0;
            cv::Size label_size =
                cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);

            int y = std::max(tl.y - 10, label_size.height);
            cv::Point label_tl(tl.x, y - label_size.height);
            cv::Point label_br(tl.x + label_size.width, y + base_line);

            cv::rectangle(result, label_tl, label_br, color, cv::FILLED);
            cv::putText(result, label, cv::Point(tl.x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        return result;
    }

private:
    std::string model_path_;

    int input_height_ = 0;
    int input_width_ = 0;

    int img_height_ = 0;
    int img_width_ = 0;
    float gain_ = 1.0f;
    int pad_top_ = 0;
    int pad_left_ = 0;

    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;

    std::unique_ptr<dxrt::InferenceEngine> ie_;
    std::unique_ptr<YOLOv5PostProcess> postprocess_;
};

}  // namespace

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN

    std::string model_path;
    std::string image_path;
    std::string video_path;
    std::string rtsp_url;
    int camera_index = -1;
    bool display = true;

    std::string app_name = "YOLOv5 Sync C++ Example";
    cxxopts::Options options(app_name, app_name + " usage");

    // Long options are aligned with the Python example; short aliases provide
    // a concise CLI: -m (model), -i (image), -v (video), -c (camera), -r (rtsp).
    options.add_options()
        ("m,model", "Input DXNN model", cxxopts::value<std::string>(model_path))
        ("i,image", "Path to input image.", cxxopts::value<std::string>(image_path))
        ("v,video", "Path to input video.", cxxopts::value<std::string>(video_path))
        ("c,camera", "Camera device index (e.g., 0).",
        cxxopts::value<int>(camera_index))
        ("r,rtsp", "RTSP stream URL (e.g., rtsp://ip:port/stream).",
        cxxopts::value<std::string>(rtsp_url))
        ("no-display", "Do not display window (still runs inference).",
        cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    auto cmd = options.parse(argc, argv);

    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!cmd.count("model")) {
        std::cerr << "[ERROR] --model is required" << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    // Mutually exclusive source selection, similar to Python example.
    int src_count = 0;
    if (cmd.count("image")) src_count++;
    if (cmd.count("video")) src_count++;
    if (cmd.count("camera")) src_count++;
    if (cmd.count("rtsp")) src_count++;

    if (src_count != 1) {
        std::cerr << "[ERROR] Please specify exactly one input source among --image/--video/--camera/--rtsp" << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    bool no_display = cmd["no-display"].as<bool>();
    display = !no_display;

    YOLOv5 model(model_path);

    if (cmd.count("image")) {
        model.image_inference(image_path, display);
    } else if (cmd.count("video")) {
        model.stream_inference(video_path, display);
    } else if (cmd.count("camera")) {
        model.stream_inference(std::to_string(camera_index), display);
    } else if (cmd.count("rtsp")) {
        model.stream_inference(rtsp_url, display);
    }

    std::cout << "\nExample completed successfully!" << std::endl;

    DXRT_TRY_CATCH_END
    return 0;
}
