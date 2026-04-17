/**
 * @file detection_visualizer.hpp
 * @brief Common detection result visualizer
 */

#ifndef DETECTION_VISUALIZER_HPP
#define DETECTION_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include "common/utility/visualization.hpp"

namespace dxapp {

/**
 * @brief Visualizer for object detection results
 */
class DetectionVisualizer : public IVisualizer<DetectionResult> {
public:
    DetectionVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<DetectionResult>& results,
                 const PreprocessContext& ctx) override {
        (void)ctx;  // Coordinates already scaled in postprocessor
        return drawDetections(frame, results, ctx, line_thickness_, font_scale_);
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        line_thickness_ = line_thickness;
        font_scale_ = font_scale;
        alpha_ = alpha;
    }

private:
    int line_thickness_{2};
    double font_scale_{0.5};
    float alpha_{0.6f};
};

}  // namespace dxapp

#endif  // DETECTION_VISUALIZER_HPP
