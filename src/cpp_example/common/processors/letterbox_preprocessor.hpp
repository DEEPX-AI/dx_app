/**
 * @file letterbox_preprocessor.hpp
 * @brief Common detection preprocessor for YOLO models
 * 
 * This preprocessor is shared by YOLOv5, YOLOv8, and other detection models.
 */

#ifndef LETTERBOX_PREPROCESSOR_HPP
#define LETTERBOX_PREPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/utility/preprocessing.hpp"

namespace dxapp {

/**
 * @brief Common preprocessor for detection models (YOLO family)
 * 
 * Performs letterbox resize and color conversion.
 * Used by YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOX, etc.
 */
class DetectionPreprocessor : public IPreprocessor {
public:
    /**
     * @brief Construct a detection preprocessor
     * @param input_width Model input width (default 640)
     * @param input_height Model input height (default 640)
     * @param color_conversion OpenCV color conversion code (default BGR2RGB)
     * @param pad_value Padding value (default 114)
     */
    DetectionPreprocessor(int input_width = 640, 
                          int input_height = 640,
                          int color_conversion = cv::COLOR_BGR2RGB,
                          int pad_value = 114)
        : input_width_(input_width), 
          input_height_(input_height),
          color_conversion_(color_conversion),
          pad_value_(pad_value) {}

    void process(const cv::Mat& input, cv::Mat& output, PreprocessContext& ctx) override {
        makeLetterboxImage(input, output, input_width_, input_height_,
                          color_conversion_, ctx, pad_value_);
    }

    int getInputWidth() const override { return input_width_; }
    int getInputHeight() const override { return input_height_; }
    int getColorConversion() const override { return color_conversion_; }

private:
    int input_width_;
    int input_height_;
    int color_conversion_;
    int pad_value_;
};

}  // namespace dxapp

#endif  // LETTERBOX_PREPROCESSOR_HPP
