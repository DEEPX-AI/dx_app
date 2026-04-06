/**
 * @file restoration_visualizer.hpp
 * @brief Image restoration result visualizer
 */

#ifndef RESTORATION_VISUALIZER_HPP
#define RESTORATION_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"

namespace dxapp {

/**
 * @brief Visualizer for image restoration results
 * 
 * Shows side-by-side: input | restored output.
 * For super-resolution, input is upscaled to match output size.
 */
class RestorationVisualizer : public IVisualizer<RestorationResult> {
public:
    RestorationVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<RestorationResult>& results,
                 const PreprocessContext& ctx) override {
        if (results.empty()) return frame.clone();

        const auto& res = results[0];
        if (res.restored_image.empty()) return frame.clone();

        cv::Mat restored = res.restored_image;
        int rh = restored.rows, rw = restored.cols;

        // Determine if super-resolution (output larger than original input)
        bool is_sr = (ctx.original_width > 0 && ctx.original_height > 0) &&
                     (rw > ctx.original_width || rh > ctx.original_height);

        cv::Mat left, right;
        if (is_sr) {
            // Upscale input to match SR output size for side-by-side
            cv::resize(frame, left, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
            right = restored;
        } else {
            // Resize restored to match input for side-by-side
            left = frame;
            cv::resize(restored, right, frame.size(), 0, 0, cv::INTER_LINEAR);
        }

        // Side-by-side canvas
        cv::Mat canvas;
        cv::hconcat(left, right, canvas);

        int font = cv::FONT_HERSHEY_SIMPLEX;
        cv::putText(canvas, "Input", cv::Point(10, 30), font, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(canvas, "Output", cv::Point(left.cols + 10, 30), font, 1.0, cv::Scalar(0, 255, 0), 2);
        return canvas;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        (void)line_thickness;
        (void)font_scale;
        (void)alpha;
    }
};

}  // namespace dxapp

#endif  // RESTORATION_VISUALIZER_HPP
