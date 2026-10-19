/**
 * @file grayscale_preprocessor.hpp
 * @brief Grayscale resize preprocessor for denoising/restoration models
 * 
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
        // The resize below is a plain cv::resize to the model's exact input
        // size, so it does not preserve aspect ratio -- a 960x540 frame going
        // into a 640x480 model is squashed by different factors in x and y.
        // Record both, the way sfa3d_bev_preprocessor does: scaleKeypoint /
        // scaleBox use the per-axis values when they are set, and fall back to
        // the uniform ctx.scale otherwise. Leaving them at 0 sent SuperPoint
        // keypoints through the uniform branch: ctx.scale = min(640/960, 480/540)
        // = 2/3, so y came back scaled by 1/(2/3) = 1.5 where the correct factor
        // is 540/480 = 1.125 -- a 1.33x vertical stretch that pushed ~20% of the
        // keypoints off the bottom of the canvas.
        ctx.scale_x = static_cast<float>(input_width_) / std::max(input.cols, 1);
        ctx.scale_y = static_cast<float>(input_height_) / std::max(input.rows, 1);
        ctx.scale = std::min(ctx.scale_x, ctx.scale_y);
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
