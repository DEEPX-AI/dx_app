/**
 * @file embedding_visualizer.hpp
 * @brief Stateful embedding comparison visualizer
 *
 * First image/frame is captured as a reference. Subsequent calls
 * produce a side-by-side comparison canvas with cosine similarity,
 * matching the demo_embedding_compare / demo_reid_compare output.
 */

#ifndef EMBEDDING_VISUALIZER_HPP
#define EMBEDDING_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace dxapp {

class EmbeddingVisualizer : public IVisualizer<EmbeddingResult> {
public:
    EmbeddingVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<EmbeddingResult>& results,
                 const PreprocessContext& ctx) override {
        (void)ctx;
        if (results.empty()) return frame.clone();

        const auto& res = results[0];

        if (!has_ref_) {
            // First call — store as reference silently (no display)
            ref_image_ = frame.clone();
            ref_embedding_ = res.embedding;
            ref_dim_ = res.dimension;
            has_ref_ = true;
            return cv::Mat();
        }

        // Subsequent calls — build comparison canvas
        float sim = cosineSimilarity(ref_embedding_, res.embedding);
        return makeComparisonCanvas(ref_image_, frame, sim, ref_dim_);
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        (void)line_thickness; (void)font_scale; (void)alpha;
    }

private:
    cv::Mat ref_image_;
    std::vector<float> ref_embedding_;
    int ref_dim_ = 0;
    bool has_ref_ = false;

    // ---- helpers ----------------------------------------------------------

    static float cosineSimilarity(const std::vector<float>& a,
                                  const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) return 0.0f;
        float dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        na = std::sqrt(na);
        nb = std::sqrt(nb);
        return (na < 1e-8f || nb < 1e-8f) ? 0.0f : dot / (na * nb);
    }

    /** Fit an image into (box_w x box_h) preserving aspect ratio. */
    static cv::Mat fitToBox(const cv::Mat& img, int box_w, int box_h) {
        float scale = std::min(static_cast<float>(box_w) / img.cols,
                               static_cast<float>(box_h) / img.rows);
        int nw = static_cast<int>(img.cols * scale);
        int nh = static_cast<int>(img.rows * scale);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(nw, nh));
        return resized;
    }

    /** Build a side-by-side comparison canvas with a minimum display size. */
    cv::Mat makeComparisonCanvas(const cv::Mat& ref, const cv::Mat& cur,
                                 float similarity, int dim) const {
        // Ensure minimum canvas size for readability
        static constexpr int MIN_W = 960;
        static constexpr int MIN_H = 640;
        int W = std::max(cur.cols, MIN_W);
        int H = std::max(cur.rows, MIN_H);
        int bar_h = 60;
        int gap = 6;
        int img_h = H - bar_h;
        int half_w = (W - gap) / 2;

        cv::Mat canvas = cv::Mat::zeros(H, W, CV_8UC3);

        // Fit both images into half-width x img_h slots
        cv::Mat r1 = fitToBox(ref, half_w, img_h);
        cv::Mat r2 = fitToBox(cur, half_w, img_h);

        // Center each in its slot
        int x1 = (half_w - r1.cols) / 2;
        int y1 = (img_h - r1.rows) / 2;
        r1.copyTo(canvas(cv::Rect(x1, y1, r1.cols, r1.rows)));

        int x2 = half_w + gap + (half_w - r2.cols) / 2;
        int y2 = (img_h - r2.rows) / 2;
        r2.copyTo(canvas(cv::Rect(x2, y2, r2.cols, r2.rows)));

        // Separator
        cv::line(canvas,
                 cv::Point(half_w + gap / 2, 0),
                 cv::Point(half_w + gap / 2, img_h),
                 cv::Scalar(80, 80, 80), 1);

        // Labels
        cv::putText(canvas, "Reference",
                    cv::Point(x1 + 4, y1 + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(canvas, "Current",
                    cv::Point(x2 + 4, y2 + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        // Bottom bar — similarity result
        cv::Mat bar_overlay = canvas.clone();
        cv::rectangle(bar_overlay,
                      cv::Point(0, img_h),
                      cv::Point(W, H),
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(bar_overlay, 0.7, canvas, 0.3, 0, canvas);

        bool match = similarity > 0.4f;
        cv::Scalar color = match ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 220);
        const char* label = match ? "SAME" : "DIFFERENT";

        std::ostringstream ss;
        ss << "Cosine Similarity: " << std::fixed << std::setprecision(4)
           << similarity << "  [" << label << "]";
        cv::putText(canvas, ss.str(),
                    cv::Point(10, img_h + 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);

        return canvas;
    }
};

}  // namespace dxapp

#endif  // EMBEDDING_VISUALIZER_HPP
