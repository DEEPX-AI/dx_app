/**
 * @file centerpose_postprocessor.hpp
 * @brief CenterPose (CenterNet-based 6-DoF pose estimation) postprocessor
 * 
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
 *   4. Decode boxes, then the pose keypoints from the hps offsets around the
 *      sub-pixel centre
 *   5. Snap each keypoint onto its hm_hp peak near the centre, when present
 *
 * Keypoints and boxes match the Python CenterPosePostprocessor and the
 * dx_postprocess CenterPosePostProcess.
 */

#ifndef CENTERPOSE_POSTPROCESSOR_HPP
#define CENTERPOSE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
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
        const dxrt::TensorPtr* hm_hp_t    = nullptr;

        if (!identifyTensors_(outputs, heatmap_t, size_t_ptr, offset_t, hps_t, hm_hp_t))
            return results;

        auto shape = (*heatmap_t)->shape();
        int C = static_cast<int>(shape[1]);
        int H = static_cast<int>(shape[2]);
        int W = static_cast<int>(shape[3]);

        const float* heatmap_raw = static_cast<const float*>((*heatmap_t)->data());
        const float* sizes   = static_cast<const float*>((*size_t_ptr)->data());
        const float* offsets = offset_t ? static_cast<const float*>((*offset_t)->data()) : nullptr;
        const float* hps     = hps_t    ? static_cast<const float*>((*hps_t)->data())    : nullptr;

        // Apply sigmoid if heatmap contains raw logits (values outside [0,1]),
        // with the Python postprocessor's test and clip.
        std::vector<float> heatmap_buf;
        const float* heatmap = heatmap_raw;
        int total = C * H * W;
        if (total > 0) {
            auto mm = std::minmax_element(heatmap_raw, heatmap_raw + total);
            if (*mm.first < -0.1f || *mm.second > 1.1f) {
                heatmap_buf.resize(total);
                for (int i = 0; i < total; ++i) {
                    float v = std::min(std::max(heatmap_raw[i], -50.0f), 50.0f);
                    heatmap_buf[i] = 1.0f / (1.0f + std::exp(-v));
                }
                heatmap = heatmap_buf.data();
            }
        }

        auto heatmap_nms = computePseudoNms_(heatmap, C, H, W);
        auto peaks       = extractTopKPeaks_(heatmap_nms, C, H, W);
        auto kp_peaks    = computeKeypointPeaks_(hm_hp_t, H, W);

        // Decode detections from peaks
        std::vector<cv::Rect2d> nms_boxes;
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

            nms_boxes.push_back(cv::Rect2d(x1, y1, bw, bh));
            fboxes.push_back({x1, y1, x2, y2});
            nms_scores.push_back(p.score);
            auto kps = decodeKeypoints_(p.y, p.x, cx, cy, H, W, hps);
            refineKeypoints_(kps, kp_peaks, p.y, p.x, bw, H, W);
            all_kps.push_back(std::move(kps));
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
            pose.box = {x1, y1, x2, y2};
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

    // Categorise outputs by channel count.
    // Tensors: heatmap (C classes, often C=1), hps (2*K ch), hm_hp (K ch),
    //          size (2 ch), offset (2 ch), hps_offset (2 ch).
    bool identifyTensors_(const dxrt::TensorPtrs& outputs,
                          const dxrt::TensorPtr*& heatmap_t,
                          const dxrt::TensorPtr*& size_t_ptr,
                          const dxrt::TensorPtr*& offset_t,
                          const dxrt::TensorPtr*& hps_t,
                          const dxrt::TensorPtr*& hm_hp_t) const {
        std::vector<const dxrt::TensorPtr*> two_ch;
        const dxrt::TensorPtr* one_ch = nullptr;
        for (auto& t : outputs) {
            auto sh = t->shape();
            if (sh.size() < 3) continue;
            int ch = static_cast<int>(sh.size() == 4 ? sh[1] : sh[0]);
            if      (ch == num_keypoints_ * 2) hps_t     = &t;
            else if (ch == num_keypoints_)     hm_hp_t   = &t;
            else if (ch == 2)                  two_ch.push_back(&t);
            else if (ch == 1)                  one_ch = &t;
            else if (ch > 2)                   heatmap_t = &t;
        }
        // For single-class models (C=1), heatmap has 1 channel
        if (!heatmap_t && one_ch) heatmap_t = one_ch;
        if (!heatmap_t) return false;

        // Rank the 2-channel tensors by mean |value| over the whole tensor, as the
        // Python postprocessor does: size first, centre offset second (the
        // keypoint offset, when present, ranks last).
        std::vector<std::pair<double, const dxrt::TensorPtr*>> ranked;
        for (auto* t : two_ch) {
            size_t n = 1;
            for (auto d : (*t)->shape()) n *= static_cast<size_t>(d);
            const float* data = static_cast<const float*>((*t)->data());
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += std::abs(data[i]);
            ranked.emplace_back(n > 0 ? s / static_cast<double>(n) : 0.0, t);
        }
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        if (ranked.size() >= 1) size_t_ptr = ranked[0].second;
        if (ranked.size() >= 2) offset_t   = ranked[1].second;
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

    // Decode num_keypoints_ keypoints for a single peak (py, px): the hps offsets
    // are relative to the sub-pixel centre (cx, cy), in feature-map cells.
    std::vector<Keypoint> decodeKeypoints_(int py, int px, float cx, float cy,
                                            int H, int W, const float* hps) const {
        std::vector<Keypoint> kps(num_keypoints_, Keypoint(0.0f, 0.0f, 1.0f));
        if (!hps) return kps;
        for (int k = 0; k < num_keypoints_; ++k) {
            float kx = cx + hps[(k * 2)     * H * W + py * W + px] * stride_;
            float ky = cy + hps[(k * 2 + 1) * H * W + py * W + px] * stride_;
            kps[k] = Keypoint(kx, ky, 1.0f);
        }
        return kps;
    }

    // 3×3 local maximum of one hm_hp plane, as the Python postprocessor's
    // _heatmap_nms_2d computes it: the max-pool starts from 0, so a cell survives
    // only if it is its neighbourhood's maximum and >= 0; other cells become 0.
    static std::vector<float> keypointPeakPlane_(const float* plane, int H, int W) {
        std::vector<float> peaks(static_cast<size_t>(H) * W, 0.0f);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float mx = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= H) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= W) continue;
                        mx = std::max(mx, plane[ny * W + nx]);
                    }
                }
                float v = plane[y * W + x];
                peaks[y * W + x] = (v == mx) ? v : 0.0f;
            }
        }
        return peaks;
    }

    // Peak planes of the hm_hp channels; empty when the model has no hm_hp output
    // or its spatial size differs from the centre heatmap.
    std::vector<std::vector<float>> computeKeypointPeaks_(const dxrt::TensorPtr* hm_hp_t,
                                                          int H, int W) const {
        std::vector<std::vector<float>> kp_peaks;
        if (!hm_hp_t) return kp_peaks;
        auto sh = (*hm_hp_t)->shape();
        int hp_c = static_cast<int>(sh[sh.size() - 3]);
        if (static_cast<int>(sh[sh.size() - 2]) != H ||
            static_cast<int>(sh[sh.size() - 1]) != W) return kp_peaks;
        const float* data = static_cast<const float*>((*hm_hp_t)->data());
        for (int k = 0; k < std::min(num_keypoints_, hp_c); ++k)
            kp_peaks.push_back(keypointPeakPlane_(data + k * H * W, H, W));
        return kp_peaks;
    }

    // Replace each regressed keypoint with the strongest peak of its hm_hp plane
    // in a square of radius max(2, bw / stride / 2) cells around the centre cell
    // (py, px), when that peak exceeds 0.1. The snapped position is the peak cell
    // times the stride. Mirrors the Python _refine_keypoints_with_hm_hp.
    void refineKeypoints_(std::vector<Keypoint>& kps,
                          const std::vector<std::vector<float>>& kp_peaks,
                          int py, int px, float bw, int H, int W) const {
        if (kp_peaks.empty()) return;
        int search_r = std::max(2, static_cast<int>(bw / stride_ / 2.0f));
        int y1 = std::max(0, py - search_r), y2 = std::min(H, py + search_r + 1);
        int x1 = std::max(0, px - search_r), x2 = std::min(W, px + search_r + 1);
        if (y1 >= y2 || x1 >= x2) return;
        for (size_t k = 0; k < kp_peaks.size() && k < kps.size(); ++k) {
            const float* plane = kp_peaks[k].data();
            float best = -std::numeric_limits<float>::infinity();
            int best_y = y1, best_x = x1;
            for (int y = y1; y < y2; ++y) {
                for (int x = x1; x < x2; ++x) {
                    if (plane[y * W + x] > best) {  // first maximum, like np.argmax
                        best = plane[y * W + x];
                        best_y = y;
                        best_x = x;
                    }
                }
            }
            if (best > 0.1f) {
                kps[k].x = static_cast<float>(best_x * stride_);
                kps[k].y = static_cast<float>(best_y * stride_);
            }
        }
    }

    // Undo the letterbox on box and keypoints and clip them to the original
    // image, [0, size - 1] (in-place), as the Python postprocessor does.
    void scaleResult_(float& x1, float& y1, float& x2, float& y2,
                      std::vector<Keypoint>& kps,
                      const PreprocessContext& ctx) const {
        float gain  = std::max(ctx.scale, 1e-6f);
        float max_x = static_cast<float>(ctx.original_width - 1);
        float max_y = static_cast<float>(ctx.original_height - 1);
        auto to_x = [&](float v) { return std::min(std::max((v - ctx.pad_x) / gain, 0.0f), max_x); };
        auto to_y = [&](float v) { return std::min(std::max((v - ctx.pad_y) / gain, 0.0f), max_y); };
        x1 = to_x(x1); y1 = to_y(y1); x2 = to_x(x2); y2 = to_y(y2);
        for (auto& kp : kps) { kp.x = to_x(kp.x); kp.y = to_y(kp.y); }
    }
};

}  // namespace dxapp

#endif  // CENTERPOSE_POSTPROCESSOR_HPP
