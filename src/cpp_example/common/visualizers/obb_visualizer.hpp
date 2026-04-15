/**
 * @file obb_visualizer.hpp
 * @brief Oriented Bounding Box (OBB) visualization
 * 
 * Draws rotated bounding boxes with class labels on images.
 */

#ifndef OBB_VISUALIZER_HPP
#define OBB_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace dxapp {

// Pre-computed color table for OBB class visualization
static const std::vector<cv::Scalar> OBB_CLASS_COLORS = {
    cv::Scalar(255, 0, 0),      // Red
    cv::Scalar(0, 255, 0),      // Green
    cv::Scalar(0, 0, 255),      // Blue
    cv::Scalar(255, 255, 0),    // Cyan
    cv::Scalar(255, 0, 255),    // Magenta
    cv::Scalar(0, 255, 255),    // Yellow
    cv::Scalar(128, 0, 128),    // Purple
    cv::Scalar(255, 165, 0),    // Orange
    cv::Scalar(0, 128, 0),      // Dark Green
    cv::Scalar(128, 128, 0),    // Olive
    cv::Scalar(0, 128, 128),    // Teal
    cv::Scalar(128, 0, 0),      // Maroon
    cv::Scalar(192, 192, 192),  // Silver
    cv::Scalar(255, 192, 203),  // Pink
    cv::Scalar(255, 215, 0),    // Gold
};

class OBBVisualizer : public IVisualizer<OBBResult> {
public:
    OBBVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<OBBResult>& results,
                 const PreprocessContext& ctx) override {
        cv::Mat result = frame.clone();
        float disp_scale = 1.0f;
        if (ctx.original_width > 0 && ctx.original_height > 0 &&
            (ctx.original_width > result.cols || ctx.original_height > result.rows)) {
            disp_scale = std::min(static_cast<float>(result.cols) / ctx.original_width,
                                  static_cast<float>(result.rows) / ctx.original_height);
        }
        const float x_off = (ctx.original_width > 0)
            ? (result.cols - ctx.original_width * disp_scale) / 2.0f : 0.0f;
        const float y_off = (ctx.original_height > 0)
            ? (result.rows - ctx.original_height * disp_scale) / 2.0f : 0.0f;

        for (const auto& obb : results) {
            cv::Scalar color = OBB_CLASS_COLORS[obb.class_id % OBB_CLASS_COLORS.size()];

            // Compute 4 corner points of the rotated box
            auto corners = obb.getCorners();
            std::vector<cv::Point> pts;
            pts.reserve(4);
            for (const auto& corner : corners) {
                pts.push_back(cv::Point(static_cast<int>(corner.x * disp_scale + x_off),
                                        static_cast<int>(corner.y * disp_scale + y_off)));
            }
            cv::polylines(result, pts, true, color, line_thickness_);

            // Draw label with background
            std::string label = obb.class_name + ": " +
                                std::to_string(static_cast<int>(obb.confidence * 100)) + "%";

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                  font_scale_, 1, &baseline);

            // Find top-left of rotated box for label placement
            int min_x = std::min({pts[0].x, pts[1].x, pts[2].x, pts[3].x});
            int min_y = std::min({pts[0].y, pts[1].y, pts[2].y, pts[3].y});
            cv::Point label_pt(min_x, min_y - 10);

            cv::Point bg_tl(label_pt.x, label_pt.y - text_size.height);
            cv::Point bg_br(label_pt.x + text_size.width, label_pt.y + baseline);
            cv::rectangle(result, bg_tl, bg_br, cv::Scalar(0, 0, 0), cv::FILLED);

            cv::putText(result, label, label_pt, cv::FONT_HERSHEY_SIMPLEX,
                        font_scale_, cv::Scalar(255, 255, 255), 1);
        }

        return result;
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

#endif  // OBB_VISUALIZER_HPP
