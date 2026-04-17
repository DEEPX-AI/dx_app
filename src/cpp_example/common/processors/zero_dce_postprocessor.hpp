/**
 * @file zero_dce_postprocessor.hpp
 * @brief Zero-DCE (Zero-Reference Deep Curve Estimation) image enhancement postprocessor
 * 
 * 
 * Supports two output formats:
 *   1. [1, 24, H, W] — 8 iterations × 3 RGB channels of curve parameter maps (α)
 *      Enhancement formula (per iteration):
 *        enhanced = enhanced + alpha * enhanced * (1 - enhanced)
 *   2. [1, 3, H, W]  — Enhanced image output directly (no curve application needed)
 * 
 * The postprocessor auto-detects the format based on channel count.
 */

#ifndef ZERO_DCE_POSTPROCESSOR_HPP
#define ZERO_DCE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace dxapp {

class ZeroDCEPostprocessor : public IPostprocessor<RestorationResult> {
public:
    /**
     * @param input_width   Model input width
     * @param input_height  Model input height
     * @param num_iterations Number of LE curve iterations (default: 8)
     */
    ZeroDCEPostprocessor(int input_width, int input_height, int num_iterations = 8)
        : input_width_(input_width), input_height_(input_height),
          num_iterations_(num_iterations) {}

    std::vector<RestorationResult> process(const dxrt::TensorPtrs& outputs,
                                            const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};

        auto output = outputs[0];
        auto shape = output->shape();
        const float* data = static_cast<const float*>(output->data());
        if (!data) return {};

        // Parse shape: expect [1, 24, H, W] or [1, 3, H, W] or [24, H, W] or [3, H, W]
        int total_ch, h, w;
        if (shape.size() == 4) {
            total_ch = static_cast<int>(shape[1]);
            h = static_cast<int>(shape[2]);
            w = static_cast<int>(shape[3]);
        } else if (shape.size() == 3) {
            total_ch = static_cast<int>(shape[0]);
            h = static_cast<int>(shape[1]);
            w = static_cast<int>(shape[2]);
        } else {
            return {};
        }

        const int num_rgb = 3;
        int hw = h * w;

        // Direct enhanced image output: [3, H, W] — use as-is
        if (total_ch == num_rgb) {
            cv::Mat restored = buildRestoredImage(data, h, w);
            resizeToOriginal(restored, ctx);
            RestorationResult result;
            result.restored_image = restored;
            result.width = restored.cols;
            result.height = restored.rows;
            return { result };
        }

        // Curve parameter output: [24, H, W] — apply LE curve iterations
        int n_iters = total_ch / num_rgb;
        if (n_iters <= 0) n_iters = num_iterations_;

        // Prepare enhanced image in CHW float [0, 1]
        // From source_image (BGR uint8, stored by SimpleResizePreprocessor)
        std::vector<float> enhanced(num_rgb * hw, 0.5f);  // fallback: 0.5

        if (!ctx.source_image.empty()) {
            cv::Mat src = ctx.source_image;
            if (src.cols != w || src.rows != h) {
                cv::resize(src, src, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
            }
            loadSourceToEnhanced(src, h, w, enhanced);
        }

        // Apply iterative LE curve: E = E + α · E · (1 - E)
        applyLECurves(data, n_iters, num_rgb, hw, enhanced);

        // Convert CHW float → HWC BGR uint8
        cv::Mat restored = buildRestoredImage(enhanced.data(), h, w);
        resizeToOriginal(restored, ctx);

        RestorationResult result;
        result.restored_image = restored;
        result.width = restored.cols;
        result.height = restored.rows;

        return { result };
    }

    std::string getModelName() const override { return "Zero-DCE"; }

private:
    // Helper: resize restored image to original dimensions if available
    static void resizeToOriginal(cv::Mat& restored, const PreprocessContext& ctx) {
        if (ctx.original_width > 0 && ctx.original_height > 0 &&
            (restored.cols != ctx.original_width || restored.rows != ctx.original_height)) {
            cv::resize(restored, restored,
                       cv::Size(ctx.original_width, ctx.original_height),
                       0, 0, cv::INTER_LINEAR);
        }
    }

    // Helper: fill enhanced CHW float [0,1] from a resized source image
    // Channel layout mirrors OpenCV: enhanced[0]=ch0(B), [1]=ch1(G), [2]=ch2(R)
    static void loadSourceToEnhanced(const cv::Mat& src, int h, int w,
                                     std::vector<float>& enhanced) {
        cv::Mat bgr_src = src;
        if (src.channels() == 1) {
            cv::cvtColor(src, bgr_src, cv::COLOR_GRAY2BGR);
        }
        cv::Mat float_img;
        bgr_src.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
        std::vector<cv::Mat> channels;
        cv::split(float_img, channels);  // [0]=B, [1]=G, [2]=R
        int hw = h * w;
        std::memcpy(&enhanced[0],      channels[0].data, hw * sizeof(float));
        std::memcpy(&enhanced[hw],     channels[1].data, hw * sizeof(float));
        std::memcpy(&enhanced[2 * hw], channels[2].data, hw * sizeof(float));
    }

    // Helper: apply iterative LE curve E = E + α·E·(1−E) in-place
    static void applyLECurves(const float* data, int n_iters, int num_rgb, int hw,
                              std::vector<float>& enhanced) {
        for (int i = 0; i < n_iters; ++i) {
            for (int c = 0; c < num_rgb; ++c) {
                int alpha_offset = (i * num_rgb + c) * hw;
                int enh_offset   = c * hw;
                for (int j = 0; j < hw; ++j) {
                    float alpha = data[alpha_offset + j];
                    float e     = enhanced[enh_offset + j];
                    enhanced[enh_offset + j] = e + alpha * e * (1.0f - e);
                }
            }
        }
    }

    // Helper: convert CHW float → HWC BGR uint8 Mat (vectorized via OpenCV)
    // Channel layout: plane0, plane1, plane2 → mapped to Vec3b(plane2, plane1, plane0)
    static cv::Mat buildRestoredImage(const float* enhanced, int h, int w) {
        int hw = h * w;
        cv::Mat plane0(h, w, CV_32F, const_cast<float*>(enhanced));
        cv::Mat plane1(h, w, CV_32F, const_cast<float*>(enhanced + hw));
        cv::Mat plane2(h, w, CV_32F, const_cast<float*>(enhanced + 2 * hw));
        cv::Mat merged;
        cv::merge(std::vector<cv::Mat>{plane2, plane1, plane0}, merged);
        cv::Mat restored;
        merged.convertTo(restored, CV_8UC3, 255.0, 0.5);
        return restored;
    }

    int input_width_;
    int input_height_;
    int num_iterations_;
};

}  // namespace dxapp

#endif  // ZERO_DCE_POSTPROCESSOR_HPP
