/**
 * @file image_source.hpp
 * @brief Image file input source implementation
 */

#ifndef DXAPP_IMAGE_SOURCE_HPP
#define DXAPP_IMAGE_SOURCE_HPP

#include "../base/i_input_source.hpp"

namespace dxapp {

/**
 * @brief Input source for static image files
 * 
 * Handles loading and providing frames from image files (jpg, png, etc.)
 */
class ImageSource : public IInputSource {
public:
    /**
     * @brief Construct an image source
     * @param path Path to the image file
     */
    explicit ImageSource(const std::string& path) 
        : path_(path), frame_delivered_(false) {
        image_ = cv::imread(path, cv::IMREAD_COLOR);
        if (!image_.empty()) {
            width_ = image_.cols;
            height_ = image_.rows;
        }
    }

    bool getFrame(cv::Mat& frame) override {
        if (image_.empty() || frame_delivered_) {
            return false;
        }
        frame = image_.clone();
        frame_delivered_ = true;
        return true;
    }

    bool isOpened() const override {
        return !image_.empty();
    }

    void release() override {
        image_.release();
    }

    InputType getType() const override {
        return InputType::IMAGE;
    }

    int getWidth() const override {
        return width_;
    }

    int getHeight() const override {
        return height_;
    }

    double getFPS() const override {
        return 0.0;  // Static image has no FPS
    }

    int getTotalFrames() const override {
        return 1;  // Single frame
    }

    std::string getDescription() const override {
        return "Image file: " + path_;
    }

    bool isLiveSource() const override {
        return false;
    }

    /**
     * @brief Reset to allow reading the image again
     */
    void reset() {
        frame_delivered_ = false;
    }

private:
    std::string path_;
    cv::Mat image_;
    int width_{0};
    int height_{0};
    bool frame_delivered_;
};

}  // namespace dxapp

#endif  // DXAPP_IMAGE_SOURCE_HPP
