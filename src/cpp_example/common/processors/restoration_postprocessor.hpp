/**
 * @file restoration_postprocessor.hpp
 * @brief Image restoration postprocessors (v3-native)
 * 
 * Supports DnCNN and similar denoising/restoration models.
 */

#ifndef RESTORATION_POSTPROCESSOR_HPP
#define RESTORATION_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>

namespace dxapp {

/**
 * @brief DnCNN image denoising postprocessor
 * 
 * Input: Single tensor [1, 1, H, W] (grayscale denoised output) or [1, C, H, W].
 * Output: RestorationResult with restored image.
 * 
 * Note: DnCNN outputs the denoised image directly.
 * Values are clamped to [0, 1] and converted to uint8.
 */
/**
 * @brief Map a model-space restoration output back onto the source geometry.
 *
 * Upscaling models (RealESRGAN) have a fixed square input and the preprocessor
 * stretches the frame into it, so the raw output carries the *model input*
 * aspect ratio (192x192 -> 768x768) instead of the source one. Undoing that
 * per-axis stretch yields `original_size * upscale_factor`, matching what the
 * tiled ESPCN path produces. Same-size restoration models (DnCNN denoising,
 * upscale factor 1) are left untouched.
 */
inline cv::Mat restoreSourceGeometry(const cv::Mat& image,
                                     const PreprocessContext& ctx,
                                     int model_in_w, int model_in_h) {
    if (image.empty() || ctx.original_width <= 0 || ctx.original_height <= 0 ||
        model_in_w <= 0 || model_in_h <= 0) {
        return image;
    }
    const double scale_x = static_cast<double>(image.cols) / model_in_w;
    const double scale_y = static_cast<double>(image.rows) / model_in_h;
    if (scale_x <= 1.0 && scale_y <= 1.0) return image;

    const int target_w = std::max(1, static_cast<int>(std::lround(ctx.original_width * scale_x)));
    const int target_h = std::max(1, static_cast<int>(std::lround(ctx.original_height * scale_y)));
    if (target_w == image.cols && target_h == image.rows) return image;

    cv::Mat resized;
    const int interp = (target_w < image.cols && target_h < image.rows)
        ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize(image, resized, cv::Size(target_w, target_h), 0, 0, interp);
    return resized;
}

class DnCNNPostprocessor : public IPostprocessor<RestorationResult> {
public:
    DnCNNPostprocessor(int input_width, int input_height)
        : input_width_(input_width), input_height_(input_height) {}

    std::vector<RestorationResult> process(const dxrt::TensorPtrs& outputs,
                                            const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};

        auto output = outputs[0];
        auto shape = output->shape();
        const float* data = static_cast<const float*>(output->data());
        if (!data) return {};

        // Determine C, H, W from shape; detect NCHW vs NHWC
        int channels, h, w;
        bool is_nhwc = false;
        if (shape.size() == 4) {        // [1, C, H, W] or [1, H, W, C]
            if (shape[3] <= 4 && shape[1] > 4) {
                // NHWC: [1, H, W, C] — last dim is small channel count
                is_nhwc = true;
                h = static_cast<int>(shape[1]);
                w = static_cast<int>(shape[2]);
                channels = static_cast<int>(shape[3]);
            } else {
                // NCHW: [1, C, H, W]
                channels = static_cast<int>(shape[1]);
                h = static_cast<int>(shape[2]);
                w = static_cast<int>(shape[3]);
            }
        } else if (shape.size() == 3) { // [C, H, W] or [1, H, W]
            channels = static_cast<int>(shape[0]);
            h = static_cast<int>(shape[1]);
            w = static_cast<int>(shape[2]);
            if (channels > 4) {
                // Probably [1, H, W] with H as first dim
                channels = 1;
                h = static_cast<int>(shape[1]);
                w = static_cast<int>(shape[2]);
            }
        } else if (shape.size() == 2) { // [H, W]
            channels = 1;
            h = static_cast<int>(shape[0]);
            w = static_cast<int>(shape[1]);
        } else {
            return {};
        }

        cv::Mat restored;
        if (channels == 1) {
            // Grayscale output
            cv::Mat float_img(h, w, CV_32FC1, const_cast<float*>(data));
            cv::Mat gray_u8;
            float_img.convertTo(gray_u8, CV_8UC1, 255.0, 0.5);
            cv::cvtColor(gray_u8, restored, cv::COLOR_GRAY2BGR);
        } else if (channels == 3) {
            int hw = h * w;
            if (is_nhwc) {
                // NHWC: data is [H*W*3] interleaved RGB
                cv::Mat rgb(h, w, CV_32FC3, const_cast<float*>(data));
                cv::Mat bgr;
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                bgr.convertTo(restored, CV_8UC3, 255.0, 0.5);
            } else {
                // NCHW: data is [R-plane, G-plane, B-plane]
                cv::Mat r_ch(h, w, CV_32FC1, const_cast<float*>(data));
                cv::Mat g_ch(h, w, CV_32FC1, const_cast<float*>(data + hw));
                cv::Mat b_ch(h, w, CV_32FC1, const_cast<float*>(data + 2 * hw));
                cv::Mat merged;
                cv::merge(std::vector<cv::Mat>{b_ch, g_ch, r_ch}, merged);
                merged.convertTo(restored, CV_8UC3, 255.0, 0.5);
            }
        } else {
            return {};
        }

        // The preprocessor stretched the frame into the square model input;
        // undo it here so the result keeps the source aspect ratio.
        restored = restoreSourceGeometry(restored, ctx, input_width_, input_height_);

        RestorationResult result;
        result.restored_image = restored;
        result.width = restored.cols;
        result.height = restored.rows;

        return { result };
    }

    std::string getModelName() const override { return "DnCNN"; }

private:
    int input_width_;
    int input_height_;
};

}  // namespace dxapp

#endif  // RESTORATION_POSTPROCESSOR_HPP
