/**
 * @file zero_dce_postprocessor.hpp
 * @brief Zero-DCE (Zero-Reference Deep Curve Estimation) image enhancement postprocessor
 * 
 * Part of DX-APP v3.0.0 refactoring.
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

        // Direct enhanced image output: [3, H, W] — use as-is
        if (total_ch == num_rgb) {
            cv::Mat restored = buildRestoredImage(
                std::vector<float>(data, data + num_rgb * h * w), h, w);
            // Resize to original dimensions if available
            if (ctx.original_width > 0 && ctx.original_height > 0 &&
                (restored.cols != ctx.original_width || restored.rows != ctx.original_height)) {
                cv::resize(restored, restored,
                           cv::Size(ctx.original_width, ctx.original_height),
                           0, 0, cv::INTER_LINEAR);
            }
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
        // From source_image (RGB uint8, stored by SimpleResizePreprocessor)
        std::vector<float> enhanced(num_rgb * h * w, 0.5f);  // fallback: 0.5

        if (!ctx.source_image.empty()) {
            cv::Mat src = ctx.source_image;
            // Resize if spatial dims don't match
            if (src.cols != w || src.rows != h) {
                cv::resize(src, src, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
            }
            loadSourceToEnhanced(src, h, w, enhanced);
        }

        // Apply iterative LE curve: E = E + α · E · (1 - E)
        applyLECurves(data, n_iters, num_rgb, h * w, enhanced);

        // Convert CHW float → HWC BGR uint8
        cv::Mat restored = buildRestoredImage(enhanced, h, w);

        // Resize to original dimensions if available
        if (ctx.original_width > 0 && ctx.original_height > 0 &&
            (restored.cols != ctx.original_width || restored.rows != ctx.original_height)) {
            cv::resize(restored, restored,
                       cv::Size(ctx.original_width, ctx.original_height),
                       0, 0, cv::INTER_LINEAR);
        }

        RestorationResult result;
        result.restored_image = restored;
        result.width = restored.cols;
        result.height = restored.rows;

        return { result };
    }

    std::string getModelName() const override { return "Zero-DCE"; }

private:
    // Helper: fill enhanced CHW float [0,1] from a resized RGB source image
    static void loadSourceToEnhanced(const cv::Mat& src, int h, int w,
                                     std::vector<float>& enhanced) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (src.channels() >= 3) {
                    const auto& px = src.at<cv::Vec3b>(y, x);
                    enhanced[0 * h * w + y * w + x] = px[0] / 255.0f;  // R
                    enhanced[1 * h * w + y * w + x] = px[1] / 255.0f;  // G
                    enhanced[2 * h * w + y * w + x] = px[2] / 255.0f;  // B
                } else {
                    float v = src.at<uchar>(y, x) / 255.0f;
                    enhanced[0 * h * w + y * w + x] = v;
                    enhanced[1 * h * w + y * w + x] = v;
                    enhanced[2 * h * w + y * w + x] = v;
                }
            }
        }
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

    // Helper: convert CHW float [0,1] to HWC BGR uint8 Mat
    static cv::Mat buildRestoredImage(const std::vector<float>& enhanced, int h, int w) {
        int hw = h * w;
        cv::Mat restored(h, w, CV_8UC3);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float r = std::max(0.0f, std::min(1.0f, enhanced[0 * hw + y * w + x]));
                float g = std::max(0.0f, std::min(1.0f, enhanced[1 * hw + y * w + x]));
                float b = std::max(0.0f, std::min(1.0f, enhanced[2 * hw + y * w + x]));
                restored.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(b * 255.0f + 0.5f),
                    static_cast<uchar>(g * 255.0f + 0.5f),
                    static_cast<uchar>(r * 255.0f + 0.5f));
            }
        }
        return restored;
    }

    int input_width_;
    int input_height_;
    int num_iterations_;
};

}  // namespace dxapp

#endif  // ZERO_DCE_POSTPROCESSOR_HPP
