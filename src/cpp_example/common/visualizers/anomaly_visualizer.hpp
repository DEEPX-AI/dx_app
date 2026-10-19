/**
 * @file anomaly_visualizer.hpp
 * @brief Overlay the anomaly response on the frame, with its relative severity
 *
 * Mirrors src/python_example/common/visualizers/anomaly_visualizer.py: JET at
 * alpha 0.5 over the original frame, and a labelled severity. The depth visualizer
 * this family used before is fully opaque (alpha 1.0) with MAGMA, which hides the
 * frame the response belongs to.
 *
 * The severity is labelled "relative" on the frame itself, and that wording is
 * deliberate in both cases it serves. EfficientAD runs as the ensemble it is --
 * teacher, student and autoencoder together -- but its published score divides by
 * q_st/q_ae quantiles fitted on the training set, which no .dxnn carries. PatchCore is
 * a single network here, because its metric needs a memory bank of training features.
 * So the number ranks frames against each other for one model set, and printing a bare
 * "score" would invite exactly the comparison it cannot support.
 */

#ifndef ANOMALY_VISUALIZER_HPP
#define ANOMALY_VISUALIZER_HPP

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/base/i_visualizer.hpp"

namespace dxapp {

class AnomalyVisualizer : public IVisualizer<AnomalyResult> {
public:
    explicit AnomalyVisualizer(float alpha = 0.5f, int colormap = cv::COLORMAP_JET)
        : alpha_(alpha), colormap_(colormap) {}

    cv::Mat draw(const cv::Mat& frame, const std::vector<AnomalyResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat output = frame.clone();
        if (results.empty() || results[0].heatmap.empty()) return output;
        const AnomalyResult& result = results[0];

        cv::Mat heatmap = result.heatmap;
        if (heatmap.rows != output.rows || heatmap.cols != output.cols) {
            cv::resize(heatmap, heatmap, output.size(), 0, 0, cv::INTER_LINEAR);
        }
        cv::Mat scaled, coloured;
        heatmap.convertTo(scaled, CV_8UC1, 255.0);
        cv::applyColorMap(scaled, coloured, colormap_);
        cv::addWeighted(output, 1.0 - alpha_, coloured, alpha_, 0.0, output);

        char label[128];
        std::snprintf(label, sizeof(label), "relative severity %.4f  (%d ch)",
                      static_cast<double>(result.score), result.channels);

        // A dark plate behind ONE stroke, not an outline pass plus a fill pass:
        // cv::getTextSize's width depends on THICKNESS, so drawing the same string
        // twice at different thicknesses desynchronises the glyph advance and the two
        // copies drift apart across the line. The Python peer had exactly that bug.
        const double scale = 0.7;
        const int thickness = 1;
        int baseline = 0;
        const cv::Size text = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale,
                                              thickness, &baseline);
        const cv::Point origin(12, 12 + text.height);
        cv::Mat plate = output.clone();
        cv::rectangle(plate,
                      cv::Point(origin.x - 6, origin.y - text.height - 6),
                      cv::Point(origin.x + text.width + 6, origin.y + baseline + 4),
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(plate, 0.5, output, 0.5, 0.0, output);
        cv::putText(output, label, origin, cv::FONT_HERSHEY_SIMPLEX, scale,
                    cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        return output;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        // Only the overlay transparency is meaningful for a heatmap: there is no box
        // to thicken, and the severity label's scale is fixed so the plate fits it.
        (void)line_thickness;
        (void)font_scale;
        alpha_ = alpha;
    }

private:
    float alpha_;
    int colormap_;
};

}  // namespace dxapp

#endif  // ANOMALY_VISUALIZER_HPP
