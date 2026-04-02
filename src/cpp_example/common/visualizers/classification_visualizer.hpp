/**
 * @file classification_visualizer.hpp
 * @brief Classification result visualizer (text overlay)
 */

#ifndef CLASSIFICATION_VISUALIZER_HPP
#define CLASSIFICATION_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include "common/utility/labels.hpp"

namespace dxapp {

class ClassificationResultVisualizer : public IVisualizer<ClassificationResult> {
public:
    ClassificationResultVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<ClassificationResult>& results,
                 const PreprocessContext& ctx) override {
        (void)ctx;

        cv::Mat output = frame.clone();
        if (results.empty()) return output;

        int w = output.cols;

        int box_w = std::min(w - 20, 400);
        int box_h = 30 + static_cast<int>(results.size()) * 25;
        cv::Mat overlay = output.clone();
        cv::rectangle(overlay, cv::Point(10, 10), cv::Point(10 + box_w, 10 + box_h),
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.5, output, 0.5, 0, output);

        int y_offset = 30;
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& res = results[i];
            std::string label = res.class_name.empty()
                ? getImageNetClassName(res.class_id)
                : res.class_name;
            std::string text = "#" + std::to_string(i + 1) + ": " + label +
                               " (" + formatConfidence(res.confidence) + ")";

            cv::putText(output, text, cv::Point(20, y_offset),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale_,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            y_offset += 25;
        }

        return output;
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
    double font_scale_{0.6};
    float alpha_{0.5f};

    std::string formatConfidence(float conf) const {
        std::ostringstream oss;
        float pct = (conf <= 1.01f) ? conf * 100.0f : conf;
        oss << std::fixed << std::setprecision(1) << pct << "%";
        return oss.str();
    }
};

}  // namespace dxapp

#endif  // CLASSIFICATION_VISUALIZER_HPP
