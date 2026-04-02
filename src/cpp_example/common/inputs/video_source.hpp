/**
 * @file video_source.hpp
 * @brief Video file input source implementation
 * 
 * Part of DX-APP v3.0.0 refactoring for independent build capability.
 */

#ifndef DXAPP_VIDEO_SOURCE_HPP
#define DXAPP_VIDEO_SOURCE_HPP

#include "../base/i_input_source.hpp"

namespace dxapp {

/**
 * @brief Input source for video files
 * 
 * Handles loading and providing frames from video files (mp4, avi, etc.)
 */
class VideoSource : public IInputSource {
public:
    /**
     * @brief Construct a video source
     * @param path Path to the video file
     */
    explicit VideoSource(const std::string& path) : path_(path) {
        capture_.open(path);
        if (capture_.isOpened()) {
            width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
            height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
            fps_ = capture_.get(cv::CAP_PROP_FPS);
            total_frames_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
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
        return InputType::VIDEO;
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
        return total_frames_;
    }

    std::string getDescription() const override {
        return "Video file: " + path_;
    }

    bool isLiveSource() const override {
        return false;
    }

    /**
     * @brief Seek to a specific frame
     * @param frame_num Frame number to seek to
     * @return true if seek was successful
     */
    bool seek(int frame_num) {
        return capture_.set(cv::CAP_PROP_POS_FRAMES, frame_num);
    }

    /**
     * @brief Get current frame position
     * @return Current frame number
     */
    int getCurrentFrame() const {
        return static_cast<int>(capture_.get(cv::CAP_PROP_POS_FRAMES));
    }

private:
    std::string path_;
    cv::VideoCapture capture_;
    int width_{0};
    int height_{0};
    double fps_{0.0};
    int total_frames_{0};
};

}  // namespace dxapp

#endif  // DXAPP_VIDEO_SOURCE_HPP
