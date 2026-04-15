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

        RestorationResult result;
        result.restored_image = restored;
        result.width = w;
        result.height = h;

        return { result };
    }

    std::string getModelName() const override { return "DnCNN"; }

private:
    int input_width_;
    int input_height_;
};

}  // namespace dxapp

#endif  // RESTORATION_POSTPROCESSOR_HPP
