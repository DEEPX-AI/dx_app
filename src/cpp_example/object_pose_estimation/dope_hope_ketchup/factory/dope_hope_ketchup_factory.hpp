/**
 * @file dope_hope_ketchup_factory.hpp
 * @brief DopeHopeKetchupFactory: 6-DoF object pose estimation (DOPE model)
 *
 * Model outputs (merged by dxrt):
 *   output: FLOAT [1, 25, H/8, W/8]
 *     channels  0–8  : 9 belief maps  (8 vertices + centroid) — conv2d_84
 *     channels  9–24 : 16 affinity fields                     — conv2d_91 (unused)
 *
 * Post-processing:
 *   Argmax peak-picking on the 9-channel belief maps → 9 image-space 2D points.
 *   The cuboid overlay is drawn by connecting these 9 predicted 2-D vertices
 *   directly (DOPE already predicts the projected cuboid corners), so NO PnP /
 *   opencv_calib3d is required. (Python example keeps cv2.solvePnP for the
 *   metric 6-DoF pose; the C++ demo is 2-D overlay only.)
 *
 *   Vertex order = FTR(0) FTL(1) FBL(2) FBR(3) RTR(4) RTL(5) RBL(6) RBR(7) Center(8)
 */

#ifndef DOPE_HOPE_KETCHUP_FACTORY_HPP
#define DOPE_HOPE_KETCHUP_FACTORY_HPP

#include <array>

#include <opencv2/imgproc.hpp>  // cv::Mat/circle/putText/line (core+imgproc; calib3d no longer needed)

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/pose_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "dope_postprocess.h"

namespace dxapp {

// ---------------------------------------------------------------------------
// Postprocessor wrapper — converts DopeResult (heatmap coords) to PoseResult
// ---------------------------------------------------------------------------
class DopePostprocessorWrapper : public IPostprocessor<PoseResult> {
public:
    DopePostprocessorWrapper(int input_w, int input_h)
        : impl_(input_w, input_h),
          hm_w_(input_w / 8), hm_h_(input_h / 8) {}

    std::vector<PoseResult> process(const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) override {
        DopeResult dr = impl_.postprocess(outputs);

        // Scale heatmap coordinates → original image coordinates
        const float sx = static_cast<float>(ctx.original_width)  / hm_w_;
        const float sy = static_cast<float>(ctx.original_height) / hm_h_;

        PoseResult pr;
        // Report the centroid (channel 8 = last) belief peak as the object
        // confidence instead of a hardcoded 1.0.
        pr.confidence = dr.peaks.empty() ? 0.f : dr.peaks.back().confidence;
        pr.keypoints.reserve(dr.peaks.size());
        for (const auto& p : dr.peaks)
            pr.keypoints.emplace_back(p.x * sx, p.y * sy, p.confidence);

        return {std::move(pr)};
    }

    std::string getModelName() const override { return "DOPE Hope-Ketchup"; }

private:
    DOPEPostProcess impl_;
    int hm_w_, hm_h_;
};

// ---------------------------------------------------------------------------
// Visualizer — connect the 9 predicted 2-D cuboid vertices directly.
// DOPE already predicts the projected cuboid corners, so the overlay needs no
// PnP / projectPoints (and therefore no opencv_calib3d). 2-D demo overlay only.
// ---------------------------------------------------------------------------
class DopeCuboidVisualizer : public IVisualizer<PoseResult> {
public:
    cv::Mat draw(const cv::Mat& image,
                 const std::vector<PoseResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat out = image.clone();
        const int W = out.cols, H = out.rows;

        // 12 cuboid edges (DOPE CuboidLineIndexes order)
        static constexpr std::array<std::pair<int,int>, 12> kEdges = {{
            {1,0},{0,3},{3,2},{2,1},   // front face
            {5,4},{4,7},{7,6},{6,5},   // rear face
            {2,6},{1,5},{3,7},{0,4},   // connecting
        }};
        // X on top face
        static constexpr std::array<std::pair<int,int>, 2> kTopX = {{
            {0,5},{1,4}
        }};

        static const cv::Scalar kColors[8] = {
            {0,0,255},{0,255,0},{255,0,0},{0,255,255},
            {255,0,255},{255,128,0},{0,255,128},{128,0,255}
        };

        for (const auto& pr : results) {
            const auto& kps = pr.keypoints;
            if (kps.size() < 9) continue;

            // --- Draw belief-peak dots ---
            for (int i = 0; i < 8; ++i) {
                cv::Point p(static_cast<int>(kps[i].x),
                            static_cast<int>(kps[i].y));
                if (p.x >= 0 && p.x < W && p.y >= 0 && p.y < H) {
                    cv::circle(out, p, 6, kColors[i], -1, cv::LINE_AA);
                    cv::putText(out, std::to_string(i), p + cv::Point(6, 0),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, kColors[i], 1);
                }
            }

            // --- Draw centroid ---
            cv::Point cp(static_cast<int>(kps[8].x),
                         static_cast<int>(kps[8].y));
            if (cp.x >= 0 && cp.x < W && cp.y >= 0 && cp.y < H) {
                cv::circle(out, cp, 8, cv::Scalar(255,255,255), -1, cv::LINE_AA);
                cv::circle(out, cp, 8, cv::Scalar(0,0,0), 2, cv::LINE_AA);
            }

            // --- Draw cuboid by connecting the predicted 2-D vertices ---
            drawEdges(out, kps, kEdges, cv::Scalar(0,255,0), W, H);   // box
            drawEdges(out, kps, kTopX,  cv::Scalar(0,255,255), W, H); // top X
        }

        cv::putText(out, "DOPE Pose (2D)", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,255), 2);
        return out;
    }

    void setParameters(int, double, float) override {}

private:
    template<typename Kps, size_t N>
    static void drawEdges(cv::Mat& out, const Kps& kps,
                          const std::array<std::pair<int,int>, N>& edges,
                          const cv::Scalar& color, int W, int H) {
        auto valid = [&](int i) {
            return kps[i].x >= 0 && kps[i].x < W &&
                   kps[i].y >= 0 && kps[i].y < H;
        };
        auto pt = [&](int i) {
            return cv::Point(static_cast<int>(kps[i].x),
                             static_cast<int>(kps[i].y));
        };
        for (const auto& e : edges)
            if (valid(e.first) && valid(e.second))
                cv::line(out, pt(e.first), pt(e.second), color, 2, cv::LINE_AA);
    }
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
class DopeHopeKetchupFactory : public IPoseFactory {
public:
    DopeHopeKetchupFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool /*is_ort_configured*/ = false) override {
        return std::make_unique<DopePostprocessorWrapper>(input_width, input_height);
    }

    VisualizerPtr<PoseResult> createVisualizer() override {
        return std::make_unique<DopeCuboidVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& /*config*/) override {}
    std::string getModelName() const override { return "DOPE Hope-Ketchup"; }
    std::string getTaskType() const override { return "object_pose_estimation"; }
};

}  // namespace dxapp

#endif  // DOPE_HOPE_KETCHUP_FACTORY_HPP
