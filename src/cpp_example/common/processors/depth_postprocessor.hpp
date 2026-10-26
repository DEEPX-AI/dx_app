/**
 * @file depth_postprocessor.hpp
 * @brief Depth estimation postprocessors (v3-native)
 * 
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
                                      const PreprocessContext& /*ctx*/) override {
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

        // Header over the tensor, used only to read min/max. Do not assign the
        // normalized expression back onto this header: that keeps data() on the
        // tensor (refcount stays unset) and the next Run overwrites the map.
        cv::Mat wrapped(h, w, CV_32FC1, const_cast<float*>(data));
        double min_val_d = 0.0;
        double max_val_d = 0.0;
        cv::minMaxLoc(wrapped, &min_val_d, &max_val_d);
        const float min_val = static_cast<float>(min_val_d);
        const float max_val = static_cast<float>(max_val_d);

        // Owned [0, 1] map. A fresh Mat is the destination so the result
        // outlives the tensor buffer.
        cv::Mat depth_map;
        const float range = max_val - min_val;
        if (range > 1e-6f) {
            depth_map = (wrapped - min_val) / range;
        } else {
            depth_map = wrapped.clone();
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
