/**
 * @file espcn_postprocessor.hpp
 * @brief ESPCN (Efficient Sub-Pixel CNN) super-resolution postprocessor
 * 
 * 
 * ESPCN outputs the upscaled Y-channel directly (NOT a noise residual).
 * Scale factor is auto-detected from output/input ratio.
 * 
 * If a BGR source image is available in PreprocessContext::source_image
 * (stored by GrayscaleResizePreprocessor with store_source=true),
 * the postprocessor restores color by:
 *   1. Converting source BGR → YCrCb
 *   2. Bicubic-upsampling Cr and Cb to match the SR output size
 *   3. Merging the SR Y with upsampled CrCb
 *   4. Converting back YCrCb → BGR
 * 
 * Without a source image, the output is a grayscale SR image.
 */

#ifndef ESPCN_POSTPROCESSOR_HPP
#define ESPCN_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>

namespace dxapp {

class ESPCNPostprocessor : public IPostprocessor<RestorationResult> {
public:
    /**
     * @param input_width   Model input width  (e.g., 50 for ESPCN-x4)
     * @param input_height  Model input height (e.g., 50 for ESPCN-x4)
     * @param scale_factor  Upscale factor (0 = auto-detect from output shape)
     */
    ESPCNPostprocessor(int input_width, int input_height, int scale_factor = 0)
        : input_width_(input_width), input_height_(input_height),
          scale_factor_(scale_factor) {}

    std::vector<RestorationResult> process(const dxrt::TensorPtrs& outputs,
                                            const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};

        auto output = outputs[0];
        auto shape = output->shape();
        const float* data = static_cast<const float*>(output->data());
        if (!data) return {};

        // Parse output shape — expect [1, 1, H_out, W_out] or [1, H_out, W_out]
        int out_h, out_w, out_c;
        if (shape.size() == 4) {
            out_c = static_cast<int>(shape[1]);
            out_h = static_cast<int>(shape[2]);
            out_w = static_cast<int>(shape[3]);
        } else if (shape.size() == 3) {
            out_c = static_cast<int>(shape[0]);
            out_h = static_cast<int>(shape[1]);
            out_w = static_cast<int>(shape[2]);
            if (out_c > 4) {
                out_c = 1;
                out_h = static_cast<int>(shape[1]);
                out_w = static_cast<int>(shape[2]);
            }
        } else if (shape.size() == 2) {
            out_c = 1;
            out_h = static_cast<int>(shape[0]);
            out_w = static_cast<int>(shape[1]);
        } else {
            return {};
        }

        // Auto-detect scale factor
        int scale = scale_factor_;
        if (scale <= 0 && input_height_ > 0) {
            scale = std::max(1, static_cast<int>(std::round(
                static_cast<float>(out_h) / input_height_)));
        }
        if (scale <= 0) scale = 2;

        // Extract Y-channel (first channel) — clamp to [0, 1] and convert to uint8
        cv::Mat sr_y(out_h, out_w, CV_8UC1);
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float v = std::max(0.0f, std::min(1.0f, data[y * out_w + x]));
                sr_y.at<uchar>(y, x) = static_cast<uchar>(v * 255.0f + 0.5f);
            }
        }

        cv::Mat restored;

        // YCbCr color restoration when source BGR image is available
        if (out_c == 1 && !ctx.source_image.empty() && ctx.source_image.channels() >= 3) {
            // Convert source BGR → YCrCb
            cv::Mat ycrcb;
            cv::cvtColor(ctx.source_image, ycrcb, cv::COLOR_BGR2YCrCb);

            // Split into Y, Cr, Cb channels
            std::vector<cv::Mat> channels;
            cv::split(ycrcb, channels);

            // Bicubic-upsample Cr and Cb to match SR output size
            cv::Mat cr_up, cb_up;
            cv::resize(channels[1], cr_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
            cv::resize(channels[2], cb_up, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);

            // Merge SR Y + upsampled Cr + upsampled Cb
            std::vector<cv::Mat> merged = {sr_y, cr_up, cb_up};
            cv::Mat ycrcb_merged;
            cv::merge(merged, ycrcb_merged);

            // Convert back to BGR
            cv::cvtColor(ycrcb_merged, restored, cv::COLOR_YCrCb2BGR);
        } else if (out_c >= 3) {
            // Multi-channel output: treat as RGB (CHW) → BGR
            int hw = out_h * out_w;
            cv::Mat float_img(out_h, out_w, CV_32FC3);
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    int idx = y * out_w + x;
                    float r = std::max(0.0f, std::min(1.0f, data[0 * hw + idx]));
                    float g = std::max(0.0f, std::min(1.0f, data[1 * hw + idx]));
                    float b = std::max(0.0f, std::min(1.0f, data[2 * hw + idx]));
                    float_img.at<cv::Vec3f>(y, x) = cv::Vec3f(b, g, r);
                }
            }
            float_img.convertTo(restored, CV_8UC3, 255.0);
        } else {
            // Grayscale-only fallback (no source image)
            cv::cvtColor(sr_y, restored, cv::COLOR_GRAY2BGR);
        }

        RestorationResult result;
        result.restored_image = restored;
        result.width = out_w;
        result.height = out_h;

        return { result };
    }

    std::string getModelName() const override { return "ESPCN"; }

private:
    int input_width_;
    int input_height_;
    int scale_factor_;
};

}  // namespace dxapp

#endif  // ESPCN_POSTPROCESSOR_HPP
