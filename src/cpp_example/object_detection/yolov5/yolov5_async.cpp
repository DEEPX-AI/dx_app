#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <common_util.hpp>
#include <condition_variable>
#include <cstdlib>
#include <cxxopts.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "yolov5_postprocess.h"

namespace {

using Clock = std::chrono::high_resolution_clock;

struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    double sum_postprocess = 0.0;
    double sum_render = 0.0;

    int infer_completed = 0;
    bool infer_first_ts_set = false;
    bool infer_last_ts_set = false;
    Clock::time_point infer_first_ts{};
    Clock::time_point infer_last_ts{};

    bool inflight_last_ts_set = false;
    Clock::time_point inflight_last_ts{};
    int inflight_current = 0;
    int inflight_max = 0;
    double inflight_time_sum = 0.0;
};

cv::Scalar get_class_color(int class_id) {
    unsigned seed = static_cast<unsigned>(class_id * 123457u + 98765u);
    unsigned b = (seed * 16807u + 3u) & 0xFFu;
    unsigned g = (seed * 48271u + 7u) & 0xFFu;
    unsigned r = (seed * 69621u + 11u) & 0xFFu;
    return cv::Scalar(static_cast<double>(b), static_cast<double>(g), static_cast<double>(r));
}

void print_async_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                                     double total_time_sec, bool display) {
    if (metrics.infer_completed == 0 || total_frames == 0) {
        std::cout << "[WARNING] No frames processed." << std::endl;
        return;
    }

    auto avg = [&](double v) { return v / static_cast<double>(metrics.infer_completed); };
    auto fps = [](double ms) { return ms > 0.0 ? 1000.0 / ms : 0.0; };

    double avg_read = avg(metrics.sum_read);
    double avg_pre = avg(metrics.sum_preprocess);
    double avg_inf = avg(metrics.sum_inference);
    double avg_post = avg(metrics.sum_postprocess);

    double inflight_window = 0.0;
    if (metrics.infer_first_ts_set && metrics.infer_last_ts_set) {
        inflight_window = std::chrono::duration<double>(metrics.infer_last_ts - metrics.infer_first_ts).count();
    }

    double infer_tp = (inflight_window > 0.0)
                          ? static_cast<double>(metrics.infer_completed) / inflight_window
                          : 0.0;
    double inflight_avg = (inflight_window > 0.0)
                              ? metrics.inflight_time_sum / inflight_window
                              : 0.0;

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
              << std::setprecision(1) << infer_tp << " FPS*" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Postprocess" << std::right
              << std::setw(8) << std::fixed << std::setprecision(2) << avg_post << " ms     "
              << std::setw(6) << std::setprecision(1) << fps(avg_post) << " FPS" << std::endl;

    if (display) {
        double avg_render = avg(metrics.sum_render);
        std::cout << " " << std::left << std::setw(15) << "Display" << std::right
                  << std::setw(8) << std::fixed << std::setprecision(2) << avg_render
                  << " ms     " << std::setw(6) << std::setprecision(1) << fps(avg_render)
                  << " FPS" << std::endl;
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " * Actual throughput via async inference" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Completed"
              << " :    " << metrics.infer_completed << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Inflight Avg"
              << " :    " << std::fixed << std::setprecision(1) << inflight_avg << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Infer Inflight Max"
              << " :      " << metrics.inflight_max << std::endl;
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

template <typename T>
class BlockingQueue {
   public:
    bool Push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return false;
        }
        queue_.push(std::move(item));
        cond_.notify_one();
        return true;
    }

    bool Pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;
        cond_.notify_all();
    }

   private:
    std::queue<T> queue_;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable cond_;
};

class YOLOv5 {
   public:
    explicit YOLOv5(const std::string& model_path)
        : model_path_(model_path), obj_threshold_(0.25f), score_threshold_(0.3f), nms_threshold_(0.45f) {
        dxrt::InferenceOption option;
        ie_ = std::make_unique<dxrt::InferenceEngine>(model_path_, option);

        if (!dxapp::common::minversionforRTandCompiler(ie_.get())) {
            std::cerr << "[ERROR] The compiled model version is not compatible with the runtime. "
                      << "Please recompile the model." << std::endl;
            std::exit(1);
        }

        auto input_shape = ie_->GetInputs().front().shape();
        input_height_ = static_cast<int>(input_shape[1]);
        input_width_ = static_cast<int>(input_shape[2]);
        input_bytes_ = ie_->GetInputSize();
        output_bytes_ = ie_->GetOutputSize();

        postprocess_ = std::make_unique<YOLOv5PostProcess>(input_width_, input_height_, obj_threshold_,
                                                           score_threshold_, nms_threshold_,
                                                           ie_->IsOrtConfigured());

        std::cout << "\n[INFO] Model loaded: " << model_path_ << std::endl;
        std::cout << "[INFO] Model input size (WxH): " << input_width_ << "x" << input_height_
                  << std::endl;
    }

    void image_inference(const std::string& image_path, bool display) {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Failed to load image: " << image_path << std::endl;
            std::exit(1);
        }

        std::cout << "\n[INFO] Input image: " << image_path << std::endl;
        std::cout << "[INFO] Image resolution (WxH): " << image.cols << "x" << image.rows
                  << std::endl;

    ProfilingMetrics metrics;
    std::atomic<bool> stop_requested{false};

    auto produce = [&, produced = false](FrameContextPtr& ctx) mutable -> bool {
            if (stop_requested.load()) {
                return false;
            }
            if (produced) {
                return false;
            }
            ctx = std::make_shared<FrameContext>(input_bytes_, output_bytes_);
            ctx->frame_bgr = image.clone();
            ctx->t_read = 0.0;
            produced = true;
            return true;
        };

        auto start = Clock::now();
        int processed = run_async_pipeline(produce, metrics, display, stop_requested);
        auto end = Clock::now();
        double total_time = std::chrono::duration<double>(end - start).count();

        if (display && processed > 0) {
            std::cout << "\n[INFO] Press any key to close the window." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        print_async_performance_summary(metrics, processed, total_time, display);
    }

    void stream_inference(const std::string& source, bool display) {
        cv::VideoCapture cap;
        const bool is_rtsp = source.rfind("rtsp://", 0) == 0;
        const bool is_camera = !is_rtsp &&
                               !source.empty() &&
                               std::all_of(source.begin(), source.end(), [](unsigned char c) {
                                   return std::isdigit(c);
                               });

        if (is_camera) {
            int index = std::stoi(source);
            cap.open(index);
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        } else {
            cap.open(source);
            if (is_rtsp) {
                cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            }
        }

        if (!cap.isOpened()) {
            std::cerr << "[ERROR] Failed to open input source: " << source << std::endl;
            std::exit(1);
        }

        const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        if (is_camera) {
            std::cout << "\n[INFO] Camera index: " << source << std::endl;
        } else if (is_rtsp) {
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

    ProfilingMetrics metrics;
    std::atomic<bool> stop_requested{false};

    auto produce = [&](FrameContextPtr& ctx) -> bool {
            if (stop_requested.load()) {
                return false;
            }

            cv::Mat frame;
            auto t_read_start = Clock::now();
            if (!cap.read(frame) || frame.empty()) {
                return false;
            }
            auto t_read_end = Clock::now();

            ctx = std::make_shared<FrameContext>(input_bytes_, output_bytes_);
            ctx->frame_bgr = std::move(frame);
            ctx->t_read = std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
            return true;
        };

        auto start = Clock::now();
        int processed = run_async_pipeline(produce, metrics, display, stop_requested);
        auto end = Clock::now();
        double total_time = std::chrono::duration<double>(end - start).count();

        if (display) {
            cv::destroyAllWindows();
        }

        cap.release();
        print_async_performance_summary(metrics, processed, total_time, display);
    }

    private:
     struct FrameContext;
     using FrameContextPtr = std::shared_ptr<FrameContext>;

     struct FrameContext {
        FrameContext(size_t input_bytes, size_t output_bytes)
            : input_buffer(input_bytes), output_buffer(output_bytes) {}

        cv::Mat frame_bgr;
        std::vector<uint8_t> input_buffer;
        std::vector<uint8_t> output_buffer;
        dxrt::TensorPtrs outputs;
        std::vector<YOLOv5Result> detections;

        double t_read = 0.0;
        double t_preprocess = 0.0;
        double t_inference = 0.0;
        double t_postprocess = 0.0;
        double t_render = 0.0;

        Clock::time_point t_run_async_start{};
        int request_id = -1;

        int img_height = 0;
        int img_width = 0;
        float gain = 1.0f;
        int pad_top = 0;
        int pad_left = 0;
    };

    cv::Mat preprocess(FrameContext& ctx) const {
        ctx.img_height = ctx.frame_bgr.rows;
        ctx.img_width = ctx.frame_bgr.cols;
        ctx.gain = std::min(static_cast<float>(input_height_) / static_cast<float>(ctx.img_height),
                            static_cast<float>(input_width_) / static_cast<float>(ctx.img_width));

        cv::Mat img_rgb;
        cv::cvtColor(ctx.frame_bgr, img_rgb, cv::COLOR_BGR2RGB);

        cv::Mat input_mat(input_height_, input_width_, CV_8UC3, ctx.input_buffer.data());
        letterbox_into(img_rgb, input_mat, ctx);
        return input_mat;
    }

    void letterbox_into(const cv::Mat& img_rgb, cv::Mat& dst, FrameContext& ctx) const {
        int new_w = static_cast<int>(std::round(ctx.img_width * ctx.gain));
        int new_h = static_cast<int>(std::round(ctx.img_height * ctx.gain));

        cv::Mat resized;
        if (img_rgb.cols != new_w || img_rgb.rows != new_h) {
            cv::resize(img_rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img_rgb;
        }

        float dw = (static_cast<float>(input_width_) - static_cast<float>(new_w)) / 2.0f;
        float dh = (static_cast<float>(input_height_) - static_cast<float>(new_h)) / 2.0f;

        ctx.pad_left = static_cast<int>(std::round(dw - 0.1f));
        ctx.pad_top = static_cast<int>(std::round(dh - 0.1f));

        dst.setTo(cv::Scalar(114, 114, 114));
        cv::Rect roi(ctx.pad_left, ctx.pad_top, resized.cols, resized.rows);
        resized.copyTo(dst(roi));
    }

    void scale_coordinates(const FrameContext& ctx, YOLOv5Result& det) const {
        auto clamp = [](float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); };
        float max_x = static_cast<float>(ctx.img_width - 1);
        float max_y = static_cast<float>(ctx.img_height - 1);

        det.box[0] = clamp((det.box[0] - static_cast<float>(ctx.pad_left)) / ctx.gain, 0.0f, max_x);
        det.box[1] = clamp((det.box[1] - static_cast<float>(ctx.pad_top)) / ctx.gain, 0.0f, max_y);
        det.box[2] = clamp((det.box[2] - static_cast<float>(ctx.pad_left)) / ctx.gain, 0.0f, max_x);
        det.box[3] = clamp((det.box[3] - static_cast<float>(ctx.pad_top)) / ctx.gain, 0.0f, max_y);
    }

    cv::Mat draw_detections(FrameContext& ctx) const {
        cv::Mat result = ctx.frame_bgr.clone();
        for (auto& det : ctx.detections) {
            scale_coordinates(ctx, det);

            cv::Point tl(static_cast<int>(det.box[0]), static_cast<int>(det.box[1]));
            cv::Point br(static_cast<int>(det.box[2]), static_cast<int>(det.box[3]));
            cv::Scalar color = get_class_color(det.class_id);
            cv::rectangle(result, tl, br, color, 2);

            std::ostringstream label_stream;
            label_stream << det.class_name << ": " << std::fixed << std::setprecision(2) << det.confidence;
            std::string label = label_stream.str();

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

    template <typename Producer>
    int run_async_pipeline(Producer&& produce_frame, ProfilingMetrics& metrics, bool display,
                           std::atomic<bool>& stop_requested) {
        BlockingQueue<FrameContextPtr> input_queue;
        BlockingQueue<FrameContextPtr> wait_queue;
        BlockingQueue<FrameContextPtr> post_queue;
        BlockingQueue<FrameContextPtr> render_queue;

        std::mutex metrics_mutex;
        std::atomic<int> processed_frames{0};

        auto request_stop = [&]() {
            bool expected = false;
            if (!stop_requested.compare_exchange_strong(expected, true)) {
                return;
            }
            input_queue.Close();
            wait_queue.Close();
            post_queue.Close();
            render_queue.Close();
        };

        auto preprocess_worker = [&]() {
            FrameContextPtr ctx;
            while (input_queue.Pop(ctx)) {
                if (!ctx) {
                    continue;
                }

                auto t0 = Clock::now();
                cv::Mat input_mat = preprocess(*ctx);
                auto t1 = Clock::now();
                ctx->t_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
                ctx->t_run_async_start = t1;

                ctx->request_id = ie_->RunAsync(input_mat.data, nullptr, ctx->output_buffer.data());
                auto submit_ts = Clock::now();

                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    if (!metrics.infer_first_ts_set) {
                        metrics.infer_first_ts = ctx->t_run_async_start;
                        metrics.infer_first_ts_set = true;
                    }

                    if (!metrics.inflight_last_ts_set) {
                        metrics.inflight_last_ts = submit_ts;
                        metrics.inflight_last_ts_set = true;
                    } else {
                        double dt = std::chrono::duration<double>(submit_ts - metrics.inflight_last_ts).count();
                        metrics.inflight_time_sum += metrics.inflight_current * dt;
                        metrics.inflight_last_ts = submit_ts;
                    }

                    metrics.inflight_current++;
                    metrics.inflight_max = std::max(metrics.inflight_max, metrics.inflight_current);
                }

                if (!wait_queue.Push(std::move(ctx))) {
                    request_stop();
                    break;
                }
            }
            wait_queue.Close();
        };

        auto wait_worker = [&]() {
            FrameContextPtr ctx;
            while (wait_queue.Pop(ctx)) {
                if (!ctx) {
                    continue;
                }

                ctx->outputs = ie_->Wait(ctx->request_id);
                auto t_done = Clock::now();
                ctx->t_inference =
                    std::chrono::duration<double, std::milli>(t_done - ctx->t_run_async_start).count();

                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics.infer_last_ts = t_done;
                    metrics.infer_last_ts_set = true;

                    if (metrics.inflight_last_ts_set) {
                        double dt =
                            std::chrono::duration<double>(t_done - metrics.inflight_last_ts).count();
                        metrics.inflight_time_sum += metrics.inflight_current * dt;
                    }
                    metrics.inflight_last_ts = t_done;
                    metrics.inflight_last_ts_set = true;

                    metrics.infer_completed++;
                    metrics.inflight_current = std::max(0, metrics.inflight_current - 1);
                }

                if (!post_queue.Push(std::move(ctx))) {
                    request_stop();
                    break;
                }
            }
            post_queue.Close();
        };

        auto postprocess_worker = [&]() {
            FrameContextPtr ctx;
            while (post_queue.Pop(ctx)) {
                if (!ctx) {
                    continue;
                }

                auto t0 = Clock::now();
                ctx->detections = postprocess_->postprocess(ctx->outputs);
                ctx->outputs.clear();
                auto t1 = Clock::now();
                ctx->t_postprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();

                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics.sum_read += ctx->t_read;
                    metrics.sum_preprocess += ctx->t_preprocess;
                    metrics.sum_inference += ctx->t_inference;
                    metrics.sum_postprocess += ctx->t_postprocess;
                }

                if (!render_queue.Push(std::move(ctx))) {
                    request_stop();
                    break;
                }
            }
            render_queue.Close();
        };

        auto render_worker = [&]() {
            FrameContextPtr ctx;
            const std::string window_name = "YOLOv5 Async Output";
            while (render_queue.Pop(ctx)) {
                if (!ctx) {
                    continue;
                }

                double render_ms = 0.0;
                if (display) {
                    auto render_start = Clock::now();
                    cv::Mat output = draw_detections(*ctx);
                    cv::imshow(window_name, output);
                    int key = cv::waitKey(1) & 0xFF;
                    auto render_end = Clock::now();
                    render_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();

                    if (key == 'q' || key == 27) {
                        request_stop();
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics.sum_render += render_ms;
                }

                processed_frames.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::thread preprocess_thread(preprocess_worker);
        std::thread wait_thread(wait_worker);
        std::thread post_thread(postprocess_worker);
        std::thread render_thread(render_worker);

        FrameContextPtr ctx;
        while (!stop_requested.load()) {
            if (!produce_frame(ctx)) {
                break;
            }
            if (!ctx) {
                continue;
            }
            if (!input_queue.Push(std::move(ctx))) {
                break;
            }
        }

        input_queue.Close();

        preprocess_thread.join();
        wait_thread.join();
        post_thread.join();
        render_thread.join();

        return processed_frames.load();
    }

    std::string model_path_;
    int input_height_ = 0;
    int input_width_ = 0;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;

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

    std::string app_name = "YOLOv5 Async C++ Example";
    cxxopts::Options options(app_name, app_name + " usage");

    options.add_options()
        ("m,model", "Input DXNN model", cxxopts::value<std::string>(model_path))
        ("i,image", "Path to input image.", cxxopts::value<std::string>(image_path))
        ("v,video", "Path to input video.", cxxopts::value<std::string>(video_path))
        ("c,camera", "Camera device index (e.g., 0).", cxxopts::value<int>(camera_index))
        ("r,rtsp", "RTSP stream URL (e.g., rtsp://ip:port/stream).", cxxopts::value<std::string>(rtsp_url))
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

    int src_count = 0;
    if (cmd.count("image")) src_count++;
    if (cmd.count("video")) src_count++;
    if (cmd.count("camera")) src_count++;
    if (cmd.count("rtsp")) src_count++;

    if (src_count != 1) {
        std::cerr << "[ERROR] Please specify exactly one input source among --image/--video/--camera/--rtsp"
                  << std::endl;
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