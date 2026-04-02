/**
 * @file restoration_postprocessor.hpp
 * @brief Image restoration postprocessors (v3-native)
 * 
 * Part of DX-APP v3.0.0 refactoring.
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

        // Determine C, H, W from shape
        int channels, h, w;
        if (shape.size() == 4) {        // [1, C, H, W]
            channels = static_cast<int>(shape[1]);
            h = static_cast<int>(shape[2]);
            w = static_cast<int>(shape[3]);
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
            cv::Mat float_img(h, w, CV_32FC1);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    float v = std::max(0.0f, std::min(1.0f, data[y * w + x]));
                    float_img.at<float>(y, x) = v;
                }
            }
            float_img.convertTo(restored, CV_8UC1, 255.0);
            cv::cvtColor(restored, restored, cv::COLOR_GRAY2BGR);
        } else if (channels == 3) {
            // RGB output → BGR
            cv::Mat float_img(h, w, CV_32FC3);
            int hw = h * w;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int idx = y * w + x;
                    float r = std::max(0.0f, std::min(1.0f, data[0 * hw + idx]));
                    float g = std::max(0.0f, std::min(1.0f, data[1 * hw + idx]));
                    float b = std::max(0.0f, std::min(1.0f, data[2 * hw + idx]));
                    float_img.at<cv::Vec3f>(y, x) = cv::Vec3f(b, g, r); // BGR
                }
            }
            float_img.convertTo(restored, CV_8UC3, 255.0);
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
