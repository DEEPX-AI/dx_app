/**
 * @file centerpose_postprocessor.hpp
 * @brief CenterPose (CenterNet-based 6-DoF pose estimation) postprocessor
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Ported from Python centerpose_postprocessor.py.
 * 
 * CenterPose uses CenterNet-style multi-head heatmap outputs with 8 keypoints.
 * Output: 6 tensors at stride 4:
 *   - heatmap:    [1, C, H/4, W/4]   center heatmap (class-wise, C >= 3)
 *   - size:       [1, 2, H/4, W/4]   width/height regression
 *   - offset:     [1, 2, H/4, W/4]   center offset
 *   - hps:        [1, 16, H/4, W/4]  keypoint offsets (8 × 2)
 *   - hps_offset: [1, 2, H/4, W/4]   keypoint offset refinement
 *   - hm_hp:      [1, 8, H/4, W/4]   keypoint heatmaps (8 keypoints)
 * 
 * Algorithm:
 *   1. Identify tensors by channel count
 *   2. Pseudo-NMS on center heatmap + keypoint heatmaps
 *   3. Top-K center detection
 *   4. Decode boxes, then decode 8 pose keypoints
 */

#ifndef CENTERPOSE_POSTPROCESSOR_HPP
#define CENTERPOSE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

class CenterPosePostprocessor : public IPostprocessor<PoseResult> {
public:
    CenterPosePostprocessor(int input_width = 512, int input_height = 512,
                            float score_threshold = 0.3f,
                            float nms_threshold = 0.5f,
                            int num_keypoints = 8,
                            int stride = 4,
                            int top_k = 100)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          num_keypoints_(num_keypoints), stride_(stride), top_k_(top_k) {}

    std::vector<PoseResult> process(const dxrt::TensorPtrs& outputs,
                                     const PreprocessContext& ctx) override {
        std::vector<PoseResult> results;
        if (outputs.size() < 3) return results;

        const dxrt::TensorPtr* heatmap_t  = nullptr;
        const dxrt::TensorPtr* size_t_ptr = nullptr;
        const dxrt::TensorPtr* offset_t   = nullptr;
        const dxrt::TensorPtr* hps_t      = nullptr;

        if (!identifyTensors_(outputs, heatmap_t, size_t_ptr, offset_t, hps_t))
            return results;

        auto shape = (*heatmap_t)->shape();
        int C = static_cast<int>(shape[1]);
        int H = static_cast<int>(shape[2]);
        int W = static_cast<int>(shape[3]);

        const float* heatmap = static_cast<const float*>((*heatmap_t)->data());
        const float* sizes   = static_cast<const float*>((*size_t_ptr)->data());
        const float* offsets = offset_t ? static_cast<const float*>((*offset_t)->data()) : nullptr;
        const float* hps     = hps_t    ? static_cast<const float*>((*hps_t)->data())    : nullptr;

        auto heatmap_nms = computePseudoNms_(heatmap, C, H, W);
        auto peaks       = extractTopKPeaks_(heatmap_nms, C, H, W);

        // Decode detections from peaks
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<std::array<float, 4>> fboxes;
        std::vector<std::vector<Keypoint>> all_kps;

        for (auto& p : peaks) {
            float off_x = offsets ? offsets[0 * H * W + p.y * W + p.x] : 0.0f;
            float off_y = offsets ? offsets[1 * H * W + p.y * W + p.x] : 0.0f;
            float w = sizes[0 * H * W + p.y * W + p.x];
            float h = sizes[1 * H * W + p.y * W + p.x];

            float cx = (p.x + off_x) * stride_;
            float cy = (p.y + off_y) * stride_;
            float bw = w * stride_, bh = h * stride_;

            float x1 = cx - bw * 0.5f, y1 = cy - bh * 0.5f;
            float x2 = cx + bw * 0.5f, y2 = cy + bh * 0.5f;

            nms_boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            fboxes.push_back({x1, y1, x2, y2});
            nms_scores.push_back(p.score);
            all_kps.push_back(decodeKeypoints_(p.y, p.x, H, W, hps));
        }

        if (nms_boxes.empty()) return results;

        std::vector<int> indices;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, indices);

        for (int idx : indices) {
            float x1 = fboxes[idx][0], y1 = fboxes[idx][1];
            float x2 = fboxes[idx][2], y2 = fboxes[idx][3];
            auto kps = all_kps[idx];
            scaleResult_(x1, y1, x2, y2, kps, ctx);

            PoseResult pose;
            pose.box = {
                std::max(0.0f, x1), std::max(0.0f, y1),
                std::min(x2, static_cast<float>(ctx.original_width)),
                std::min(y2, static_cast<float>(ctx.original_height))
            };
            pose.confidence = nms_scores[idx];
            pose.keypoints  = std::move(kps);
            results.push_back(pose);
        }
        return results;
    }

    std::string getModelName() const override { return "CenterPose"; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    int num_keypoints_;
    int stride_;
    int top_k_;

    struct Peak_ { float score; int cls, y, x; };

    // Categorise outputs: heatmap (C>2), hps (2*K ch), 2-ch tensors → size/offset.
    bool identifyTensors_(const dxrt::TensorPtrs& outputs,
                          const dxrt::TensorPtr*& heatmap_t,
                          const dxrt::TensorPtr*& size_t_ptr,
                          const dxrt::TensorPtr*& offset_t,
                          const dxrt::TensorPtr*& hps_t) const {
        std::vector<const dxrt::TensorPtr*> two_ch;
        for (auto& t : outputs) {
            auto sh = t->shape();
            if (sh.size() < 3) continue;
            int ch = static_cast<int>(sh.size() == 4 ? sh[1] : sh[0]);
            if      (ch == num_keypoints_ * 2) hps_t     = &t;
            else if (ch == num_keypoints_)     { /* hm_hp – not used in decode */ }
            else if (ch == 2)                  two_ch.push_back(&t);
            else if (ch > 2)                   heatmap_t = &t;
        }
        if (!heatmap_t) return false;

        auto sh = (*heatmap_t)->shape();
        int H = static_cast<int>(sh[2]), W = static_cast<int>(sh[3]);

        if (two_ch.size() >= 2) {
            const float* d0 = static_cast<const float*>((*two_ch[0])->data());
            const float* d1 = static_cast<const float*>((*two_ch[1])->data());
            float s0 = 0, s1 = 0;
            int n = std::min(2 * H * W, 1000);
            for (int i = 0; i < n; ++i) { s0 += std::abs(d0[i]); s1 += std::abs(d1[i]); }
            if (s0 >= s1) { size_t_ptr = two_ch[0]; offset_t = two_ch[1]; }
            else           { size_t_ptr = two_ch[1]; offset_t = two_ch[0]; }
        } else if (two_ch.size() == 1) {
            size_t_ptr = two_ch[0];
        }
        return size_t_ptr != nullptr;
    }

    // 3×3 max-pool pseudo-NMS on a flat C×H×W heatmap.
    std::vector<float> computePseudoNms_(const float* heatmap, int C, int H, int W) const {
        std::vector<float> nms(C * H * W, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float val = heatmap[c * H * W + y * W + x];
                    float mx = val;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int ny = y + dy, nx = x + dx;
                            if (ny >= 0 && ny < H && nx >= 0 && nx < W)
                                mx = std::max(mx, heatmap[c * H * W + ny * W + nx]);
                        }
                    }
                    nms[c * H * W + y * W + x] = (val >= mx) ? val : 0.0f;
                }
            }
        }
        return nms;
    }

    // Collect above-threshold peaks and return the top-K sorted by score.
    std::vector<Peak_> extractTopKPeaks_(const std::vector<float>& nms,
                                          int C, int H, int W) const {
        std::vector<Peak_> peaks;
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float s = nms[c * H * W + y * W + x];
                    if (s > score_threshold_) peaks.push_back({s, c, y, x});
                }
        std::sort(peaks.begin(), peaks.end(),
                  [](const Peak_& a, const Peak_& b) { return a.score > b.score; });
        if ((int)peaks.size() > top_k_) peaks.resize(top_k_);
        return peaks;
    }

    // Decode num_keypoints_ keypoints from hps tensor for a single peak (py, px).
    std::vector<Keypoint> decodeKeypoints_(int py, int px, int H, int W,
                                            const float* hps) const {
        std::vector<Keypoint> kps(num_keypoints_);
        if (!hps) return kps;
        for (int k = 0; k < num_keypoints_; ++k) {
            float kx = (px + hps[(k * 2)     * H * W + py * W + px]) * stride_;
            float ky = (py + hps[(k * 2 + 1) * H * W + py * W + px]) * stride_;
            kps[k] = Keypoint(kx, ky, 1.0f);
        }
        return kps;
    }

    // Scale box and keypoints to original-image coordinates (in-place).
    void scaleResult_(float& x1, float& y1, float& x2, float& y2,
                      std::vector<Keypoint>& kps,
                      const PreprocessContext& ctx) const {
        if (ctx.pad_x == 0 && ctx.pad_y == 0) {
            float sx = static_cast<float>(ctx.original_width)  / input_width_;
            float sy = static_cast<float>(ctx.original_height) / input_height_;
            x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy;
            for (auto& kp : kps) { kp.x *= sx; kp.y *= sy; }
        } else {
            x1 = (x1 - ctx.pad_x) / ctx.scale;
            y1 = (y1 - ctx.pad_y) / ctx.scale;
            x2 = (x2 - ctx.pad_x) / ctx.scale;
            y2 = (y2 - ctx.pad_y) / ctx.scale;
            for (auto& kp : kps) {
                kp.x = (kp.x - ctx.pad_x) / ctx.scale;
                kp.y = (kp.y - ctx.pad_y) / ctx.scale;
            }
        }
    }
};

}  // namespace dxapp

#endif  // CENTERPOSE_POSTPROCESSOR_HPP
