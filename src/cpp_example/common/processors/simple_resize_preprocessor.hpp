/**
 * @file simple_resize_preprocessor.hpp
 * @brief Simple resize preprocessor (no letterbox, no aspect ratio preservation)
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Used by SSD, BiseNet, FastDepth, and other models that expect direct resize.
 */

#ifndef SIMPLE_RESIZE_PREPROCESSOR_HPP
#define SIMPLE_RESIZE_PREPROCESSOR_HPP

#include "common/base/i_processor.hpp"

namespace dxapp {

/**
 * @brief Preprocessor that directly resizes input without letterbox padding.
 * Sets pad_x=0, pad_y=0 to indicate simple resize mode.
 */
class SimpleResizePreprocessor : public IPreprocessor {
public:
    SimpleResizePreprocessor(int input_width = 300, int input_height = 300,
                             int color_conversion = cv::COLOR_BGR2RGB,
                             bool store_source = false)
        : input_width_(input_width), input_height_(input_height),
          color_conversion_(color_conversion), store_source_(store_source) {}

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

        // Convert color space
        cv::Mat converted;
        if (color_conversion_ >= 0) {
            cv::cvtColor(input, converted, color_conversion_);
        } else {
            converted = input;
        }

        // Direct resize (no letterbox)
        cv::resize(converted, output, cv::Size(input_width_, input_height_),
                   0, 0, cv::INTER_LINEAR);

        // Store resized image for postprocessors that need it (e.g., Zero-DCE)
        if (store_source_) {
            ctx.source_image = output.clone();
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
};

}  // namespace dxapp

#endif  // SIMPLE_RESIZE_PREPROCESSOR_HPP
