/**
 * @file camera_source.hpp
 * @brief Camera input source implementation
 */

#ifndef DXAPP_CAMERA_SOURCE_HPP
#define DXAPP_CAMERA_SOURCE_HPP

#include "../base/i_input_source.hpp"

namespace dxapp {

/**
 * @brief Input source for camera devices
 * 
 * Handles capturing frames from camera devices (USB webcams, etc.)
 */
class CameraSource : public IInputSource {
public:
    /**
     * @brief Construct a camera source
     * @param device_id Camera device index (e.g., 0 for /dev/video0)
     * @param width Desired capture width (0 for default)
     * @param height Desired capture height (0 for default)
     * @param fps Desired capture FPS (0 for default)
     */
    explicit CameraSource(int device_id, int width = 0, int height = 0, double fps = 0.0) 
        : device_id_(device_id) {
        capture_.open(device_id);
        
        if (capture_.isOpened()) {
            // Set desired resolution if specified
            if (width > 0) capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
            if (height > 0) capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            if (fps > 0) capture_.set(cv::CAP_PROP_FPS, fps);
            
            // Read actual values
            width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
            height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
            fps_ = capture_.get(cv::CAP_PROP_FPS);
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
        return InputType::CAMERA;
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
        return "Camera index: " + std::to_string(device_id_);
    }

    bool isLiveSource() const override {
        return true;
    }

private:
    int device_id_;
    cv::VideoCapture capture_;
    int width_{0};
    int height_{0};
    double fps_{30.0};
};

}  // namespace dxapp

#endif  // DXAPP_CAMERA_SOURCE_HPP
