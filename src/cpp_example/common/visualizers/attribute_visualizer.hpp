/**
 * @file attribute_visualizer.hpp
 * @brief Person/Face attribute recognition result visualizer
 *
 * Displays activated attributes as a text list overlay.
 */

#ifndef ATTRIBUTE_VISUALIZER_HPP
#define ATTRIBUTE_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"

#include <iomanip>
#include <sstream>

namespace dxapp {

class AttributeVisualizer : public IVisualizer<ClassificationResult> {
public:
    AttributeVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<ClassificationResult>& results,
                 const PreprocessContext& ctx) override {
        (void)ctx;
        cv::Mat output = frame.clone();
        if (results.empty()) return output;

        int w = output.cols;
        int line_h = 22;
        int pad = 8;
        int box_w = std::min(w - 20, 320);
        int box_h = pad * 2 + (static_cast<int>(results.size()) + 1) * line_h;

        // Semi-transparent background
        cv::Mat overlay = output.clone();
        cv::rectangle(overlay, cv::Point(10, 10),
                      cv::Point(10 + box_w, 10 + box_h),
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.55, output, 0.45, 0, output);

        // Title
        int y = 10 + pad + line_h;
        cv::putText(output, "[Person Attributes]", cv::Point(20, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255),
                    1, cv::LINE_AA);
        y += line_h;

        // Attribute list
        for (const auto& res : results) {
            std::ostringstream oss;
            oss << res.class_name << ": "
                << std::fixed << std::setprecision(0)
                << (res.confidence * 100.0f) << "%";

            cv::Scalar color = (res.confidence > 0.7f)
                ? cv::Scalar(0, 255, 0)
                : cv::Scalar(0, 200, 200);

            cv::putText(output, oss.str(), cv::Point(25, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv::LINE_AA);
            y += line_h;
        }

        return output;
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

#endif  // ATTRIBUTE_VISUALIZER_HPP
