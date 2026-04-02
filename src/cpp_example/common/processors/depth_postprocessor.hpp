/**
 * @file depth_postprocessor.hpp
 * @brief Depth estimation postprocessors (v3-native)
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Supports FastDepth and similar single-output depth models.
 */

#ifndef DEPTH_POSTPROCESSOR_HPP
#define DEPTH_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>

namespace dxapp {

/**
 * @brief FastDepth / MobileNet-based depth estimation postprocessor
 * 
 * Input: Single tensor [1, 1, H, W] or [1, H, W] containing raw depth values.
 * Output: DepthResult with depth_map, normalized to [0, 1] range.
 */
class FastDepthPostprocessor : public IPostprocessor<DepthResult> {
public:
    FastDepthPostprocessor(int input_width, int input_height)
        : input_width_(input_width), input_height_(input_height) {}

    std::vector<DepthResult> process(const dxrt::TensorPtrs& outputs,
                                      const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};

        auto output = outputs[0];
        auto shape = output->shape();
        const float* data = static_cast<const float*>(output->data());
        if (!data) return {};

        // Determine H, W from shape
        int h, w;
        if (shape.size() == 4) {        // [1, 1, H, W]
            h = static_cast<int>(shape[2]);
            w = static_cast<int>(shape[3]);
        } else if (shape.size() == 3) { // [1, H, W]
            h = static_cast<int>(shape[1]);
            w = static_cast<int>(shape[2]);
        } else if (shape.size() == 2) { // [H, W]
            h = static_cast<int>(shape[0]);
            w = static_cast<int>(shape[1]);
        } else {
            return {};
        }

        // Create depth map and find min/max
        cv::Mat depth_map(h, w, CV_32FC1);
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        int total = h * w;
        for (int i = 0; i < total; ++i) {
            float v = data[i];
            depth_map.at<float>(i / w, i % w) = v;
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }

        // Normalize to [0, 1]
        float range = max_val - min_val;
        if (range > 1e-6f) {
            depth_map = (depth_map - min_val) / range;
        }

        DepthResult result;
        result.depth_map = depth_map;
        result.width = w;
        result.height = h;
        result.min_depth = min_val;
        result.max_depth = max_val;

        return { result };
    }

    std::string getModelName() const override { return "FastDepth"; }

private:
    int input_width_;
    int input_height_;
};

}  // namespace dxapp

#endif  // DEPTH_POSTPROCESSOR_HPP
