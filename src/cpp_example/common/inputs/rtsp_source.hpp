/**
 * @file rtsp_source.hpp
 * @brief RTSP stream input source implementation
 */

#ifndef DXAPP_RTSP_SOURCE_HPP
#define DXAPP_RTSP_SOURCE_HPP

#include "../base/i_input_source.hpp"

namespace dxapp {

/**
 * @brief Input source for RTSP streams
 * 
 * Handles capturing frames from RTSP network streams
 */
class RTSPSource : public IInputSource {
public:
    /**
     * @brief Construct an RTSP source
     * @param url RTSP stream URL
     */
    explicit RTSPSource(const std::string& url) : url_(url) {
        capture_.open(url);
        
        if (capture_.isOpened()) {
            width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
            height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
            fps_ = capture_.get(cv::CAP_PROP_FPS);
            
            // Some RTSP streams don't report FPS correctly
            if (fps_ <= 0) fps_ = 30.0;
        }
    }

    bool getFrame(cv::Mat& frame) override {
        if (!capture_.isOpened()) {
            return false;
        }
        capture_ >> frame;
        return !frame.empty();
    }

    bool isOpened() const override {
        return capture_.isOpened();
    }

    void release() override {
        capture_.release();
    }

    InputType getType() const override {
        return InputType::RTSP;
    }

    int getWidth() const override {
        return width_;
    }

    int getHeight() const override {
        return height_;
    }

    double getFPS() const override {
        return fps_;
    }

    int getTotalFrames() const override {
        return -1;  // Infinite stream
    }

    std::string getDescription() const override {
        return "RTSP URL: " + url_;
    }

    bool isLiveSource() const override {
        return true;
    }

    /**
     * @brief Attempt to reconnect to the stream
     * @return true if reconnection was successful
     */
    bool reconnect() {
        capture_.release();
        capture_.open(url_);
        return capture_.isOpened();
    }

private:
    std::string url_;
    cv::VideoCapture capture_;
    int width_{0};
    int height_{0};
    double fps_{30.0};
};

}  // namespace dxapp

#endif  // DXAPP_RTSP_SOURCE_HPP
