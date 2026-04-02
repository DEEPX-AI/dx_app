/**
 * @file grayscale_preprocessor.hpp
 * @brief Grayscale resize preprocessor for denoising/restoration models
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Used by DnCNN and other grayscale-input models.
 */

#ifndef GRAYSCALE_PREPROCESSOR_HPP
#define GRAYSCALE_PREPROCESSOR_HPP

#include "common/base/i_processor.hpp"

namespace dxapp {

/**
 * @brief Preprocessor that converts to grayscale and resizes.
 * Sets pad_x=0, pad_y=0 (no letterbox).
 */
class GrayscaleResizePreprocessor : public IPreprocessor {
public:
    GrayscaleResizePreprocessor(int input_width = 50, int input_height = 50,
                                bool store_source = false)
        : input_width_(input_width), input_height_(input_height),
          store_source_(store_source) {}

    void process(const cv::Mat& input, cv::Mat& output, PreprocessContext& ctx) override {
        ctx.original_width = input.cols;
        ctx.original_height = input.rows;
        ctx.input_width = input_width_;
        ctx.input_height = input_height_;
        ctx.scale = std::min(
            static_cast<float>(input_width_) / input.cols,
            static_cast<float>(input_height_) / input.rows
        );
        ctx.pad_x = 0;
        ctx.pad_y = 0;

        // Store BGR source image for postprocessors that need color info (e.g., ESPCN)
        if (store_source_) {
            ctx.source_image = input.clone();
        }

        // Convert to grayscale
        cv::Mat gray;
        if (input.channels() == 3) {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Resize to model input size
        cv::resize(gray, output, cv::Size(input_width_, input_height_),
                   0, 0, cv::INTER_LINEAR);
    }

    int getInputWidth() const override { return input_width_; }
    int getInputHeight() const override { return input_height_; }
    int getColorConversion() const override { return cv::COLOR_BGR2GRAY; }

private:
    int input_width_;
    int input_height_;
    bool store_source_;
};

}  // namespace dxapp

#endif  // GRAYSCALE_PREPROCESSOR_HPP
