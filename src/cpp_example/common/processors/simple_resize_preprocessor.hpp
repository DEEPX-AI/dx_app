/**
 * @file simple_resize_preprocessor.hpp
 * @brief Simple resize preprocessor (no letterbox, no aspect ratio preservation)
 * 
 * Used by SSD, BiseNet, FastDepth, and other models that expect direct resize.
 */

#ifndef SIMPLE_RESIZE_PREPROCESSOR_HPP
#define SIMPLE_RESIZE_PREPROCESSOR_HPP

#include <array>

#include "common/base/i_processor.hpp"

namespace dxapp {

/**
 * @brief Preprocessor that directly resizes input without letterbox padding.
 * Sets pad_x=0, pad_y=0 to indicate simple resize mode.
 *
 * Optional mean subtraction (e.g. RetinaFace BGR mean=[104,117,123]):
 *   pass mean={104,117,123}, output_float=true, color_conversion=-1 (keep BGR).
 */
class SimpleResizePreprocessor : public IPreprocessor {
public:
    SimpleResizePreprocessor(int input_width = 300, int input_height = 300,
                             int color_conversion = cv::COLOR_BGR2RGB,
                             bool store_source = false,
                             std::array<float, 3> mean = {0.f, 0.f, 0.f},
                             bool output_float = false,
                             bool normalize_float = false)
        : input_width_(input_width), input_height_(input_height),
          color_conversion_(color_conversion), store_source_(store_source),
          mean_(mean), output_float_(output_float), normalize_float_(normalize_float) {}

    void process(const cv::Mat& input, cv::Mat& output, PreprocessContext& ctx) override {
        ctx.original_width = input.cols;
        ctx.original_height = input.rows;
        ctx.input_width = input_width_;
        ctx.input_height = input_height_;
        ctx.scale_x = static_cast<float>(input_width_) / input.cols;
        ctx.scale_y = static_cast<float>(input_height_) / input.rows;
        ctx.scale = std::min(ctx.scale_x, ctx.scale_y);
        ctx.pad_x = 0;
        ctx.pad_y = 0;

        // Convert color space (pass -1 to skip conversion and keep BGR)
        cv::Mat converted;
        if (color_conversion_ >= 0) {
            cv::cvtColor(input, converted, color_conversion_);
        } else {
            converted = input;
        }

        // Direct resize (no letterbox)
        cv::Mat resized;
        cv::resize(converted, resized, cv::Size(input_width_, input_height_),
                   0, 0, cv::INTER_LINEAR);

        // Store resized image for postprocessors that need it (e.g., Zero-DCE)
        if (store_source_) {
            ctx.source_image = resized.clone();
        }

        // Optional float32 output with mean subtraction
        if (output_float_) {
            resized.convertTo(output, CV_32FC3);
            if (normalize_float_) {
                output /= 255.0f;
            }
            if (mean_[0] != 0.f || mean_[1] != 0.f || mean_[2] != 0.f) {
                output -= cv::Scalar(mean_[0], mean_[1], mean_[2]);
            }
        } else {
            output = resized;
        }
    }

    int getInputWidth() const override { return input_width_; }
    int getInputHeight() const override { return input_height_; }
    int getColorConversion() const override { return color_conversion_; }

private:
    int input_width_;
    int input_height_;
    int color_conversion_;
    bool store_source_;
    std::array<float, 3> mean_;
    bool output_float_;
    bool normalize_float_;
};

}  // namespace dxapp

#endif  // SIMPLE_RESIZE_PREPROCESSOR_HPP
