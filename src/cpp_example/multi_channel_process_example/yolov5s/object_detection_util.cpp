#include "object_detection_util.h"

#include <dxrt/inference_engine.h>
#include <dxrt/tensor.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <vector>

using namespace std::chrono;

// 전역 GUI 뮤텍스 (멀티스레드 환경에서 OpenCV GUI 함수 동기화용)
static std::mutex gui_mutex;
// 후처리 뮤텍스

namespace {
// Helper: determines source type from path (same logic as async example's inputs)
ChannelProcessor::SourceType detectSourceType(const std::string &path) {
    if (path.rfind("rtsp://", 0) == 0) return ChannelProcessor::SourceType::RTSP;
    if (path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0)
        return ChannelProcessor::SourceType::HTTP;
    if (path.rfind("/dev/video", 0) == 0) return ChannelProcessor::SourceType::CAMERA;
    // Try integer camera index
    bool allDigits = !path.empty() && std::all_of(path.begin(), path.end(), ::isdigit);
    if (allDigits) return ChannelProcessor::SourceType::CAMERA;

    // Simple extension checks
    auto dot = path.find_last_of('.');
    if (dot != std::string::npos) {
        auto ext = path.substr(dot + 1);
        for (auto &c : ext) c = ::tolower(c);
        if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "webp")
            return ChannelProcessor::SourceType::IMAGE;
        if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv")
            return ChannelProcessor::SourceType::VIDEO;
    }
    return ChannelProcessor::SourceType::UNKNOWN;
}

// Letterbox helper aligned with yolov5_async.cpp
static void make_letterbox_image(const cv::Mat &image, cv::Mat &preprocessed_image,
                                 const int color_space, std::vector<int> &pad_xy) {
    int input_width = preprocessed_image.cols;
    int input_height = preprocessed_image.rows;
    int letterbox_pad_x = pad_xy[0];
    int letterbox_pad_y = pad_xy[1];

    cv::Mat resized_image;
    if (letterbox_pad_x == 0 && letterbox_pad_y == 0) {
        cv::resize(image, resized_image, cv::Size(input_width, input_height));
        cv::cvtColor(resized_image, preprocessed_image, color_space);
        return;
    }

    int new_width = input_width - letterbox_pad_x * 2;
    int new_height = input_height - letterbox_pad_y * 2;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));
    cv::cvtColor(resized_image, resized_image, color_space);
    int top = letterbox_pad_y;
    int bottom = letterbox_pad_y;
    int left = letterbox_pad_x;
    int right = letterbox_pad_x;

    cv::copyMakeBorder(resized_image, preprocessed_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114));
}

static void scale_coordinates(YOLOv5Result &detection, const std::vector<int> &pad, float scale) {
    detection.box[0] = (detection.box[0] - static_cast<float>(pad[0])) / scale;
    detection.box[1] = (detection.box[1] - static_cast<float>(pad[1])) / scale;
    detection.box[2] = (detection.box[2] - static_cast<float>(pad[0])) / scale;
    detection.box[3] = (detection.box[3] - static_cast<float>(pad[1])) / scale;
};

static cv::Scalar get_class_color(int class_id) {
    std::srand(class_id);
    int b = std::rand() % 256;
    int g = std::rand() % 256;
    int r = std::rand() % 256;
    return cv::Scalar(b, g, r);
};

// Draw helper aligned with yolov5_async.cpp
static cv::Mat draw_detections(const cv::Mat &frame, std::vector<YOLOv5Result> &detections,
                               const std::vector<int> &pad_xy, const float letterbox_scale) {
    cv::Mat result = frame.clone();
    for (auto &d : detections) {
        scale_coordinates(d, pad_xy, letterbox_scale);
        cv::Scalar box_color = get_class_color(d.class_id);
        cv::Point2f tl(d.box[0], d.box[1]);
        cv::Point2f br(d.box[2], d.box[3]);
        cv::rectangle(result, tl, br, box_color, 2);
        std::string conf_text =
            d.class_name + ": " + std::to_string(static_cast<int>(d.confidence * 100)) + "%";
        int baseline = 0;
        auto text_size = cv::getTextSize(conf_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point text_pos(d.box[0], d.box[1] - 10);
        cv::rectangle(result, cv::Point(text_pos.x, text_pos.y - text_size.height),
                      cv::Point(text_pos.x + text_size.width, text_pos.y + baseline),
                      cv::Scalar(0, 0, 0), -1);
        cv::putText(result, conf_text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
    return result;
}
}  // namespace

// Minimal implementation wiring variables in header
class ChannelProcessorImpl {
   public:
    explicit ChannelProcessorImpl(ChannelProcessor *self) : self_(self) {}

    bool initializeIE(const std::shared_ptr<dxrt::InferenceEngine> &ie) {
        try {
            self_->ie_ = ie;
            // Validate runtime/compiler compatibility if available
            // Note: using minimal info similar to yolov5_async
            auto input_shape = self_->ie_->GetInputs().front().shape();
            int input_height = static_cast<int>(input_shape[1]);
            int input_width = static_cast<int>(input_shape[2]);
            self_->input_size_ = cv::Size(input_width, input_height);
            self_->input_length_ = self_->ie_->GetInputSize();
            self_->output_length_ = self_->ie_->GetOutputSize();
            self_->input_buffers_ = std::vector<std::vector<uint8_t>>(self_->buffer_size_);
            self_->output_buffers_ = std::vector<std::vector<uint8_t>>(self_->buffer_size_);
            self_->frame_buffers_ = std::vector<cv::Mat>(self_->buffer_size_);
            for (auto &b : self_->input_buffers_) b = std::vector<uint8_t>(self_->input_length_);
            for (auto &b : self_->output_buffers_) b = std::vector<uint8_t>(self_->output_length_);

            self_->post_processor_ = std::make_unique<YOLOv5PostProcess>(
                input_width, input_height, self_->object_threshold_, self_->score_threshold_,
                self_->nms_threshold_, self_->ie_->IsOrtConfigured());
            return true;
        } catch (const std::exception &e) {
            std::cerr << "IE init exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool openSource(const std::string &path) {
        std::lock_guard<std::mutex> lk(self_->frame_lock_);
        self_->source_path_ = path;
        self_->source_type_ = detectSourceType(path);

        if (self_->source_type_ == ChannelProcessor::SourceType::IMAGE) {
            cv::Mat img = cv::imread(path);
            if (img.empty()) return false;
            cv::resize(img, img, self_->display_size_);
            self_->source_size_ = img.size();
            self_->frame_size_ = img.size();
            self_->display_frame_ = img;
            self_->tmp_frame_ = img;
            self_->is_opened_ = true;
            self_->total_frames_ = 1;
            self_->fps_ = 0.0;
            return true;
        }

        cap_ = std::make_unique<cv::VideoCapture>();
        bool ok = false;
        if (self_->source_type_ == ChannelProcessor::SourceType::CAMERA &&
            std::all_of(path.begin(), path.end(), ::isdigit)) {
            ok = cap_->open(std::stoi(path), cv::CAP_ANY);
            ok = cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);
            ok = cap_->set(cv::CAP_PROP_FRAME_WIDTH, self_->display_size_.width);
            ok = cap_->set(cv::CAP_PROP_FRAME_HEIGHT, self_->display_size_.height);
        } else {
            ok = cap_->open(path);
        }
        if (!ok) return false;

        self_->is_opened_ = true;
        self_->fps_ = cap_->get(cv::CAP_PROP_FPS);
        self_->total_frames_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_COUNT));
        int w = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
        self_->source_size_ = cv::Size(w, h);
        self_->frame_size_ = self_->source_size_;
        return true;
    }

    bool fetchFrame(cv::Mat &frame) {
        std::lock_guard<std::mutex> lk(self_->frame_lock_);
        if (!self_->is_opened_) return false;
        if (self_->source_type_ == ChannelProcessor::SourceType::IMAGE) {
            frame = self_->tmp_frame_.clone();
            return !frame.empty();
        }
        if (!cap_) return false;
        cv::Mat origin_frame;
        auto ret = cap_->read(origin_frame);
        int i = 0;
        while (origin_frame.empty()) {
            cap_->set(cv::CAP_PROP_POS_FRAMES, 0);
            // try video capture
            ret = cap_->read(origin_frame);
            i++;
            if (!ret && i > 30) break;
        }
        cv::resize(origin_frame, frame, self_->display_size_);
        return ret;
    }

    bool fetchFrameToBuffer(int frame_index) {
        std::lock_guard<std::mutex> lk(self_->frame_lock_);
        if (!self_->is_opened_) return false;
        if (self_->source_type_ == ChannelProcessor::SourceType::IMAGE) {
            self_->frame_buffers_[frame_index] = self_->tmp_frame_.clone();
            return !self_->frame_buffers_[frame_index].empty();
        }
        if (!cap_) return false;
        cv::Mat origin_frame;
        auto ret = cap_->read(origin_frame);
        int i = 0;
        while (origin_frame.empty()) {
            cap_->set(cv::CAP_PROP_POS_FRAMES, 0);
            // try video capture
            ret = cap_->read(origin_frame);
            i++;
            if (!ret && i > 30) break;
        }
        cv::resize(origin_frame, self_->frame_buffers_[frame_index], self_->display_size_);
        return ret;
    }

    bool preprocess(const cv::Mat &src, std::vector<uint8_t> &inputBuffer, std::vector<int> &padXY,
                    float &scale_factor) {
        if (self_->input_size_.width <= 0 || self_->input_size_.height <= 0) return false;

        // Compute scale and padding like yolov5_async
        scale_factor = std::min(self_->input_size_.width / static_cast<float>(src.cols),
                                self_->input_size_.height / static_cast<float>(src.rows));
        int letterbox_pad_x =
            std::max(0.f, (self_->input_size_.width - src.cols * scale_factor) / 2);
        int letterbox_pad_y =
            std::max(0.f, (self_->input_size_.height - src.rows * scale_factor) / 2);
        padXY = {letterbox_pad_x, letterbox_pad_y};

        cv::Mat preprocessed_image(self_->input_size_.height, self_->input_size_.width, CV_8UC3,
                                   inputBuffer.data());
        make_letterbox_image(src, preprocessed_image, cv::COLOR_BGR2RGB, padXY);
        return true;
    }

    bool runInferenceAsync(const std::vector<uint8_t> &inputBuffer,
                           std::vector<uint8_t> &outputBuffer, int &request_id) {
        if (!self_->ie_) return false;
        request_id = self_->ie_->RunAsync(const_cast<uint8_t *>(inputBuffer.data()), nullptr,
                                          outputBuffer.data());
        ++self_->processed_count_;
        return request_id >= 0;
    }

    std::vector<YOLOv5Result> waitAndPostprocess(int request_id) {
        if (!self_->post_processor_) return {};
        auto outputs = self_->ie_->Wait(request_id);
        return self_->post_processor_->postprocess(outputs);
    }

    void renderAndShow(const cv::Mat &original, std::vector<YOLOv5Result> &dets,
                       const std::string &winName, const std::vector<int> &padXY,
                       float scale_factor) {
        self_->display_frame_ = draw_detections(original, dets, padXY, scale_factor);

        // GUI 함수들을 스레드 안전하게 호출
        std::lock_guard<std::mutex> lock(gui_mutex);
        cv::imshow(winName, self_->display_frame_);
    }

    void renderAndQStack(const cv::Mat &original, std::vector<YOLOv5Result> &dets,
                         const std::vector<int> &padXY, float scale_factor) {
        self_->display_frame_ = draw_detections(original, dets, padXY, scale_factor);

        // GUI 함수들을 스레드 안전하게 호출
        std::lock_guard<std::mutex> lock(gui_mutex);
        // cv::imshow(winName, self_->display_frame_);
    }

    void pushRequestToQueue(int request_id, int frame_index, const std::vector<int> &padXY,
                            float scale_factor, const std::string &winName) {
        std::lock_guard<std::mutex> lock(self_->request_queue_mutex_);
        ChannelProcessor::RequestData req_data;
        req_data.request_id = request_id;
        req_data.frame_index = frame_index;
        req_data.padXY = padXY;
        req_data.scale_factor = scale_factor;
        req_data.winName = winName;
        self_->request_queue_.push(req_data);
    }

    bool popRequestFromQueue(ChannelProcessor::RequestData &req_data) {
        std::lock_guard<std::mutex> lock(self_->request_queue_mutex_);
        if (self_->request_queue_.empty()) {
            return false;
        }
        req_data = self_->request_queue_.front();
        self_->request_queue_.pop();
        return true;
    }

   private:
    ChannelProcessor *self_;
    std::unique_ptr<cv::VideoCapture> cap_;
};

// Public API wrapper functions (optional): provide minimal usage from outside
// Since header declares only members, we expose helper free functions below to exercise the class.

ChannelRunner makeChannel(const std::shared_ptr<dxrt::InferenceEngine> &ie,
                          const std::string &source, const cv::Size &displaySize,
                          const cv::Point &winPos) {
    ChannelRunner runner;
    runner.self = std::make_shared<ChannelProcessor>();
    runner.impl = std::make_shared<ChannelProcessorImpl>(runner.self.get());
    runner.self->setDisplaySize(displaySize);
    runner.self->setWindowPosition(winPos);
    if (!runner.impl->initializeIE(ie)) {
        std::cerr << "Channel init failed for model" << std::endl;
    }
    if (!runner.impl->openSource(source)) {
        std::cerr << "Failed opening source: " << source << std::endl;
    }
    return runner;
}

bool processOnce(ChannelRunner &runner) {
    int nextInputIndex = runner.self->getNextInputIndex();
    int nextFrameIndex = runner.self->getNextFrameIndex();

    if (!runner.impl->fetchFrameToBuffer(nextFrameIndex)) return false;

    auto &inBuf = runner.self->input_buffers_[nextInputIndex];
    auto &outBuf = runner.self->output_buffers_[nextInputIndex];
    const cv::Mat &frame = runner.self->frame_buffers_[nextFrameIndex];

    std::vector<int> padXY{0, 0};
    float scale_factor = 1.f;
    if (!runner.impl->preprocess(frame, inBuf, padXY, scale_factor)) return false;
    int req_id = -1;
    runner.impl->runInferenceAsync(inBuf, outBuf, req_id);

    // request ID를 큐에 추가
    std::string winName =
        runner.self->getSourcePath().empty() ? "channel" : runner.self->getSourcePath();
    runner.impl->pushRequestToQueue(req_id, nextFrameIndex, padXY, scale_factor, winName);

    return true;
}

bool renderOnce(ChannelRunner &runner) {
    ChannelProcessor::RequestData req_data;
    if (!runner.impl->popRequestFromQueue(req_data)) {
        // 큐가 비어있을 때는 잠시 대기 후 계속 진행
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return true;
    }

    // 추론 결과 대기 및 후처리
    auto dets = runner.impl->waitAndPostprocess(req_data.request_id);

    // 버퍼에서 frame 가져와서 렌더링
    const cv::Mat &frame = runner.self->frame_buffers_[req_data.frame_index];
    runner.impl->renderAndShow(frame, dets, req_data.winName, req_data.padXY,
                               req_data.scale_factor);

    return true;
}
