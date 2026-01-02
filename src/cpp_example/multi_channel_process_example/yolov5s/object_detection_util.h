#pragma once

#include <dxrt/dxrt_api.h>
#include <dxrt/inference_engine.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <vector>

#include "yolov5_postprocess.h"

/**
 * @brief Universal channel module that automatically handles different input sources and inference
 * recieved frames
 *
 * Supports:
 * - Image files (.jpg, .png, .bmp, etc.)
 * - Video files (.mp4, .avi, .mov, etc.)
 * - Camera devices (/dev/video0, 0, 1, etc.)
 * - RTSP streams (rtsp://...)
 * - HTTP streams (http://...)
 */
class ChannelProcessor {
   public:
    enum class SourceType { IMAGE, VIDEO, CAMERA, RTSP, HTTP, UNKNOWN };
    cv::Size getDisplaySize() { return display_size_; };
    cv::Point getWindowPosition() { return window_pos_; };
    std::string getSourcePath() { return source_path_; };
    int getNextInputIndex() { return current_input_index_++ % buffer_size_; };
    int getNextOutputIndex() { return current_output_index_++ % buffer_size_; };
    int getNextFrameIndex() { return current_frame_index_++ % buffer_size_; };

    void setDisplaySize(const cv::Size display_size) { display_size_ = display_size; };
    void setWindowPosition(const cv::Point window_pos) { window_pos_ = window_pos; };
    friend class ChannelProcessorImpl;  // allow impl access

    // For Inference
    std::shared_ptr<dxrt::InferenceEngine> ie_;
    cv::Size input_size_{0, 0};
    uint64_t input_length_{0};
    uint64_t output_length_{0};

    // For PostProcess
    std::unique_ptr<YOLOv5PostProcess> post_processor_;
    cv::Size2f padded_size_{0.f, 0.f};
    cv::Size2f scale_ratio_{1.f, 1.f};
    float object_threshold_{0.3f};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    uint64_t processed_count_{0};

    // For Frame Feeding
    std::mutex frame_lock_;
    std::string source_path_;
    SourceType source_type_{SourceType::UNKNOWN};
    cv::Size source_size_{0, 0};

    // For Queue Buffer
    std::mutex queue_lock_;
    int buffer_size_{10};
    std::condition_variable cv_;
    std::vector<std::vector<uint8_t>> input_buffers_;
    std::vector<std::vector<uint8_t>> output_buffers_;
    std::vector<cv::Mat> frame_buffers_;
    int current_input_index_{0};
    int current_output_index_{0};
    int current_frame_index_{0};

    // For Request ID Queue
    struct RequestData {
        int request_id;
        int frame_index;
        std::vector<int> padXY;
        float scale_factor;
        std::string winName;
    };
    std::queue<RequestData> request_queue_;
    std::mutex request_queue_mutex_;

    // For display result
    cv::Size display_size_{0, 0};
    cv::Point window_pos_{0, 0};
    cv::Mat tmp_frame_;
    cv::Mat display_frame_;

    // For image sources
    int loop_count_{-1};
    int current_loop_{0};
    double target_fps_{0.0};
    std::chrono::high_resolution_clock::time_point last_frame_time_;

    // Cached properties
    int total_frames_{0};
    double fps_{0.0};
    cv::Size frame_size_{0, 0};
    bool is_opened_{false};
};

// Minimal public API to drive multi-channel demo
class ChannelProcessorImpl;  // forward declaration
struct ChannelRunner {
    std::shared_ptr<ChannelProcessorImpl> impl;
    std::shared_ptr<ChannelProcessor> self;
};

ChannelRunner makeChannel(const std::shared_ptr<dxrt::InferenceEngine> &ie,
                          const std::string &source, const cv::Size &displaySize,
                          const cv::Point &winPos);
bool processOnce(ChannelRunner &runner);
bool renderOnce(ChannelRunner &runner);
