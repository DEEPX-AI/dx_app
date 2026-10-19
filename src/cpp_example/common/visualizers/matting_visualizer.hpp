/**
 * @file matting_visualizer.hpp
 * @brief Alpha matte visualisation: original, matte, and the composite it enables.
 *
 * The C++ counterpart of common/visualizers/matting_visualizer.py.
 *
 * Why this exists rather than reusing SemanticSegmentationVisualizer: that class reads
 * SegmentationResult::mask as CLASS IDS and paints it with a 19-colour Cityscapes
 * palette. Handing it a matting result renders the thresholded 0/1 map in two arbitrary
 * palette colours and throws the matte away -- the continuous opacity that is the only
 * thing matting produces. The picture looks like a segmentation, so nothing about it
 * reads as broken.
 *
 * Three panels:
 *   original   what went in.
 *   alpha      the matte, 0-255 greyscale. Soft edges (hair, fingers, motion blur) are
 *              the point of matting and are visible only here.
 *   composite  the foreground over a checkerboard, which is what the matte is FOR. A
 *              checkerboard rather than a flat colour because it makes partial opacity
 *              legible instead of guessable.
 */

#ifndef MATTING_VISUALIZER_HPP
#define MATTING_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace dxapp {

/// The transparency backdrop. Shared with the parity test, hence a free function.
inline cv::Mat mattingCheckerboard(int height, int width, int square = 16,
                                   int light = 160, int dark = 110) {
    cv::Mat board(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        auto* row = board.ptr<cv::Vec3b>(y);
        const int yb = y / square;
        for (int x = 0; x < width; ++x) {
            const unsigned char v = static_cast<unsigned char>(
                ((yb + x / square) % 2 == 0) ? light : dark);
            row[x] = cv::Vec3b(v, v, v);
        }
    }
    return board;
}

class MattingVisualizer : public IVisualizer<SegmentationResult> {
public:
    explicit MattingVisualizer(int checker_size = 16, bool show_panels = true)
        : square_(checker_size), show_panels_(show_panels) {}

    /// Foreground over the checkerboard: a*F + (1-a)*B, the matting equation.
    cv::Mat composite(const cv::Mat& frame, const cv::Mat& alpha) const {
        const cv::Mat back = mattingCheckerboard(frame.rows, frame.cols, square_);
        cv::Mat out(frame.size(), CV_8UC3);
        for (int y = 0; y < frame.rows; ++y) {
            const auto* f = frame.ptr<cv::Vec3b>(y);
            const auto* b = back.ptr<cv::Vec3b>(y);
            const float* a = alpha.ptr<float>(y);
            auto* o = out.ptr<cv::Vec3b>(y);
            for (int x = 0; x < frame.cols; ++x) {
                const float w = a[x];
                for (int c = 0; c < 3; ++c) {
                    o[x][c] = cv::saturate_cast<unsigned char>(
                        f[x][c] * w + b[x][c] * (1.0f - w));
                }
            }
        }
        return out;
    }

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<SegmentationResult>& results,
                 const PreprocessContext& ctx) override {
        (void)ctx;
        if (results.empty()) return frame.clone();

        const cv::Mat alpha = alphaOf(frame, results[0]);
        cv::Mat comp = composite(frame, alpha);
        if (!show_panels_) return comp;

        cv::Mat matte8, matte;
        alpha.convertTo(matte8, CV_8U, 255.0);
        cv::cvtColor(matte8, matte, cv::COLOR_GRAY2BGR);

        cv::Mat original = frame.clone();
        caption(original, "original");
        caption(matte, "alpha matte");
        caption(comp, "composite");

        // A 2px divider keeps the panels readable when the images are pale.
        cv::Mat div(frame.rows, 2, CV_8UC3, cv::Scalar(60, 60, 60));
        cv::Mat out;
        cv::hconcat(std::vector<cv::Mat>{original, div, matte, div, comp}, out);
        return out;
    }

    void setParameters(int line_thickness = 2, double font_scale = 0.5,
                       float alpha = 0.6f) override {
        (void)line_thickness; (void)font_scale; (void)alpha;
    }

private:
    /// The matte at frame resolution, float32 in [0,1].
    ///
    /// Falls back to the thresholded `mask` only when `alpha` is empty, so a
    /// postprocessor that produces no matte still renders something honest (a hard cut)
    /// rather than crashing.
    cv::Mat alphaOf(const cv::Mat& frame, const SegmentationResult& r) const {
        cv::Mat alpha;
        if (!r.alpha.empty()) {
            r.alpha.convertTo(alpha, CV_32F);
        } else if (!r.mask.empty() && r.width > 0 && r.height > 0) {
            cv::Mat m(r.height, r.width, CV_32S,
                      const_cast<int*>(r.mask.data()));
            m.convertTo(alpha, CV_32F);
        } else {
            return cv::Mat(frame.rows, frame.cols, CV_32F, cv::Scalar(1.0));
        }
        if (alpha.rows != frame.rows || alpha.cols != frame.cols) {
            cv::resize(alpha, alpha, cv::Size(frame.cols, frame.rows), 0, 0,
                       cv::INTER_LINEAR);
        }
        cv::min(alpha, 1.0, alpha);
        cv::max(alpha, 0.0, alpha);
        return alpha;
    }

    static void caption(cv::Mat& img, const std::string& text) {
        cv::putText(img, text, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.66,
                    cv::Scalar(20, 20, 20), 4, cv::LINE_AA);
        cv::putText(img, text, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.66,
                    cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    }

    int square_;
    bool show_panels_;
};

}  // namespace dxapp

#endif  // MATTING_VISUALIZER_HPP
