/**
 * @file embedding_visualizer.hpp
 * @brief Embedding/feature extraction result visualizer
 */

#ifndef EMBEDDING_VISUALIZER_HPP
#define EMBEDDING_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include <iomanip>
#include <sstream>

namespace dxapp {

/**
 * @brief Visualizer for embedding results
 * 
 * Displays the embedding dimension and first few values as text overlay
 * on the input frame. Since embedding results are feature vectors,
 * visualization is primarily informational.
 */
class EmbeddingVisualizer : public IVisualizer<EmbeddingResult> {
public:
    EmbeddingVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<EmbeddingResult>& results,
                 const PreprocessContext& ctx) override {
        cv::Mat display = frame.clone();

        if (results.empty()) return display;

        const auto& res = results[0];

        // Display embedding info
        std::stringstream ss;
        ss << "Embedding dim: " << res.dimension;
        cv::putText(display, ss.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale_, cv::Scalar(0, 255, 0),
                    line_thickness_, cv::LINE_AA);

        // Show first few values
        int show_count = std::min(static_cast<int>(res.embedding.size()), 8);
        std::stringstream vals;
        vals << std::fixed << std::setprecision(4) << "[";
        for (int i = 0; i < show_count; ++i) {
            if (i > 0) vals << ", ";
            vals << res.embedding[i];
        }
        if (show_count < res.dimension) vals << ", ...";
        vals << "]";

        cv::putText(display, vals.str(), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.8, cv::Scalar(0, 200, 255),
                    line_thickness_, cv::LINE_AA);

        // Show L2 norm
        float norm = 0.0f;
        for (float v : res.embedding) norm += v * v;
        norm = std::sqrt(norm);
        std::stringstream norm_ss;
        norm_ss << std::fixed << std::setprecision(4) << "L2 norm: " << norm;
        cv::putText(display, norm_ss.str(), cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale_, cv::Scalar(255, 200, 0),
                    line_thickness_, cv::LINE_AA);

        return display;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        line_thickness_ = line_thickness;
        font_scale_ = font_scale;
        alpha_ = alpha;
    }

private:
    int line_thickness_ = 2;
    double font_scale_ = 0.6;
    float alpha_ = 0.6f;
};

}  // namespace dxapp

#endif  // EMBEDDING_VISUALIZER_HPP
