/**
 * @file depth_visualizer.hpp
 * @brief Depth estimation result visualizer
 */

#ifndef DEPTH_VISUALIZER_HPP
#define DEPTH_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"

namespace dxapp {

/**
 * @brief Visualizer for depth estimation results
 * 
 * Applies a colormap (JET by default) to the depth map
 * and blends it with the original image.
 */
class DepthVisualizer : public IVisualizer<DepthResult> {
public:
    DepthVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<DepthResult>& results,
                 const PreprocessContext& ctx) override {
        if (results.empty()) return frame.clone();

        const auto& depth = results[0];
        if (depth.depth_map.empty()) return frame.clone();

        // Convert normalized depth [0,1] to uint8 for colormap
        cv::Mat depth_u8;
        depth.depth_map.convertTo(depth_u8, CV_8UC1, 255.0);

        // Apply colormap
        cv::Mat colored_depth;
        cv::applyColorMap(depth_u8, colored_depth, cv::COLORMAP_MAGMA);

        // Resize to match frame size
        cv::Mat resized_depth;
        cv::resize(colored_depth, resized_depth, frame.size(), 0, 0, cv::INTER_LINEAR);

        // alpha_ is 1.0 by default (fully opaque depth) and no caller overrides
        // it, so the blend below would be an identity pass over the whole frame.
        // Skip it — the result is bit-for-bit the same image.
        if (alpha_ >= 1.0f) {
            return resized_depth;
        }

        // Blend with original frame
        cv::Mat output;
        cv::addWeighted(frame, 1.0 - alpha_, resized_depth, alpha_, 0, output);

        return output;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        (void)line_thickness;
        (void)font_scale;
        alpha_ = alpha;
    }

private:
    float alpha_{1.0f};
};

}  // namespace dxapp

#endif  // DEPTH_VISUALIZER_HPP
