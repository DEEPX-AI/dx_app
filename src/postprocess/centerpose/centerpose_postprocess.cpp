// Copyright (C) 2018- DEEPX Ltd. All rights reserved.
/**
 * @file centerpose_postprocess.cpp
 * @brief CenterPose 6-DoF object pose estimation postprocessor
 *
 * Algorithm:
 *   1. Auto-identify tensors by channel count and mean magnitude.
 *   2. 3×3 max-pool pseudo-NMS on center heatmap.
 *   3. Extract top-K peaks above score_threshold.
 *   4. Decode center (x,y), bbox (w/h), and 8 corner keypoints.
 *   5. IoU-NMS on decoded bounding boxes.
 *
 * Coordinate output: model-input pixel space [0, input_w] × [0, input_h].
 */

#include "centerpose_postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

CenterPosePostProcess::CenterPosePostProcess(int input_w, int input_h,
                                             float score_threshold,
                                             float nms_threshold,
                                             int   num_keypoints)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      num_keypoints_(num_keypoints) {}

CenterPosePostProcess::CenterPosePostProcess()
    : input_width_(512), input_height_(512),
      score_threshold_(0.3f), nms_threshold_(0.5f),
      num_keypoints_(8) {}

// ─────────────────────────────────────────────────────────────────────────────
// Heatmap pseudo-NMS: keep pixel only if it equals its 3×3 neighbourhood max
// ─────────────────────────────────────────────────────────────────────────────

// Compute 3×3 max-pool buffer for a single [H×W] feature plane
static std::vector<float> compute_maxpool_plane_3x3(const float* plane, int H, int W) {
    std::vector<float> maxpool(H * W, -1e9f);

    auto apply_offset = [&](int dy, int dx) {
        for (int y = 0; y < H; ++y) {
            int sy = y + dy;
            if (sy < 0 || sy >= H) continue;
            for (int x = 0; x < W; ++x) {
                int sx = x + dx;
                if (sx < 0 || sx >= W) continue;
                maxpool[y * W + x] = std::max(maxpool[y * W + x], plane[sy * W + sx]);
            }
        }
    };

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            apply_offset(dy, dx);
        }
    }
    return maxpool;
}

void CenterPosePostProcess::heatmap_nms_inplace(std::vector<float>& hm,
                                                int C, int H, int W) {
    for (int c = 0; c < C; ++c) {
        float* plane = hm.data() + c * H * W;
        // Delegate 3×3 max-pool to helper, then zero out non-peak pixels
        auto maxpool = compute_maxpool_plane_3x3(plane, H, W);
        for (int i = 0; i < H * W; ++i) {
            if (plane[i] != maxpool[i]) plane[i] = 0.0f;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// postprocess – helpers
// ─────────────────────────────────────────────────────────────────────────────

// Identify and assign CenterPose output tensors by channel count and mean magnitude
static void assign_output_tensors(
    const dxrt::TensorPtrs& outputs, int K,
    dxrt::Tensor*& hm_tensor,
    dxrt::Tensor*& wh_tensor,
    dxrt::Tensor*& reg_tensor,
    dxrt::Tensor*& hps_tensor,
    dxrt::Tensor*& hmhp_tensor) {
    int K2 = K * 2;
    std::vector<dxrt::Tensor*> two_ch_tensors;
    std::vector<dxrt::Tensor*> multi_ch_tensors;

    for (const auto& t : outputs) {
        auto sh = t->shape();
        if (sh.size() < 3) continue;
        int C = static_cast<int>(sh[sh.size() - 3]);
        if (C == 2) two_ch_tensors.push_back(t.get());
        else        multi_ch_tensors.push_back(t.get());
    }

    for (dxrt::Tensor* t : multi_ch_tensors) {
        auto sh = t->shape();
        int C = static_cast<int>(sh[sh.size() - 3]);
        if      (C == K2 && hps_tensor  == nullptr) hps_tensor  = t;
        else if (C == K  && hmhp_tensor == nullptr) hmhp_tensor = t;
        else if (hm_tensor == nullptr)              hm_tensor   = t;
    }
    if (hm_tensor == nullptr && !multi_ch_tensors.empty()) {
        for (dxrt::Tensor* t : multi_ch_tensors) {
            if (t != hps_tensor && t != hmhp_tensor) { hm_tensor = t; break; }
        }
    }

    if (!two_ch_tensors.empty()) {
        auto mean_abs = [](dxrt::Tensor* t) -> float {
            const float* data = static_cast<const float*>(t->data());
            size_t n = 1;
            for (auto d : t->shape()) n *= static_cast<size_t>(d);
            float s = 0.0f;
            for (size_t i = 0; i < n; ++i) s += std::abs(data[i]);
            return n > 0 ? s / static_cast<float>(n) : 0.0f;
        };
        std::sort(two_ch_tensors.begin(), two_ch_tensors.end(),
                  [&](dxrt::Tensor* a, dxrt::Tensor* b) {
                      return mean_abs(a) > mean_abs(b);
                  });
        if (two_ch_tensors.size() >= 1) wh_tensor  = two_ch_tensors[0];
        if (two_ch_tensors.size() >= 2) reg_tensor = two_ch_tensors[1];
    }
}

// Decode K keypoints for a single peak from the hps offset map
static std::vector<float> decode_peak_keypoints(
    const float* hps_data, int K, int K2, int HW, int W,
    int ys, int xs, float cx, float cy, float stride_f) {
    std::vector<float> lm(static_cast<size_t>(K) * 3, 0.0f);
    if (hps_data == nullptr) return lm;
    for (int ki = 0; ki < K; ++ki) {
        int dx_ch = ki * 2;
        int dy_ch = ki * 2 + 1;
        float kp_x = cx;
        float kp_y = cy;
        if (dx_ch < K2) kp_x += hps_data[dx_ch * HW + ys * W + xs] * stride_f;
        if (dy_ch < K2) kp_y += hps_data[dy_ch * HW + ys * W + xs] * stride_f;
        lm[ki * 3]     = kp_x;
        lm[ki * 3 + 1] = kp_y;
        lm[ki * 3 + 2] = 1.0f;
    }
    return lm;
}

// ─────────────────────────────────────────────────────────────────────────────
// postprocess
// ─────────────────────────────────────────────────────────────────────────────

std::vector<CenterPoseResult> CenterPosePostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {

    if (outputs.size() < 4) return {};

    // ── 1. Identify tensors by channel count ────────────────────────────────
    dxrt::Tensor* hm_tensor   = nullptr;  // center heatmap   [C, H, W]
    dxrt::Tensor* wh_tensor   = nullptr;  // bbox size        [2, H, W]
    dxrt::Tensor* reg_tensor  = nullptr;  // center offset    [2, H, W]
    dxrt::Tensor* hps_tensor  = nullptr;  // keypoint offsets [K*2, H, W]
    dxrt::Tensor* hmhp_tensor = nullptr;  // kp heatmaps      [K, H, W]

    int K2 = num_keypoints_ * 2;
    int K  = num_keypoints_;

    // Delegate tensor identification to helper
    assign_output_tensors(outputs, K,
                          hm_tensor, wh_tensor, reg_tensor,
                          hps_tensor, hmhp_tensor);

    if (hm_tensor == nullptr) return {};

    // ── 2. Extract feature map dimensions ──────────────────────────────────
    auto hm_sh = hm_tensor->shape();
    int C = static_cast<int>(hm_sh[hm_sh.size() - 3]);
    int H = static_cast<int>(hm_sh[hm_sh.size() - 2]);
    int W = static_cast<int>(hm_sh[hm_sh.size() - 1]);
    int HW = H * W;

    // Copy heatmap for NMS in-place modification
    size_t hm_total = static_cast<size_t>(C) * H * W;
    std::vector<float> hm_nms(static_cast<const float*>(hm_tensor->data()),
                               static_cast<const float*>(hm_tensor->data()) + hm_total);

    // ── 3. Heatmap pseudo-NMS ───────────────────────────────────────────────
    heatmap_nms_inplace(hm_nms, C, H, W);

    // ── 4. Find top-K peaks ─────────────────────────────────────────────────
    // Flatten and find top-K by score
    int total_cells = C * HW;
    std::vector<int> all_indices(total_cells);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    int actual_top_k = std::min(top_k_, total_cells);
    std::partial_sort(all_indices.begin(), all_indices.begin() + actual_top_k,
                      all_indices.end(), [&](int a, int b) {
                          return hm_nms[a] > hm_nms[b];
                      });

    // ── 5. Decode each peak ─────────────────────────────────────────────────
    const float* wh_data  = (wh_tensor  != nullptr) ? static_cast<const float*>(wh_tensor->data())  : nullptr;
    const float* reg_data = (reg_tensor != nullptr) ? static_cast<const float*>(reg_tensor->data()) : nullptr;
    const float* hps_data = (hps_tensor != nullptr) ? static_cast<const float*>(hps_tensor->data()) : nullptr;

    float stride_f = static_cast<float>(stride_);

    std::vector<CenterPoseResult> candidates;
    candidates.reserve(actual_top_k);

    for (int k = 0; k < actual_top_k; ++k) {
        int flat_idx = all_indices[k];
        float score = hm_nms[flat_idx];
        if (score < score_threshold_) break;  // sorted descending, so stop here

        int cls      = flat_idx / HW;
        int spatial  = flat_idx % HW;
        int ys       = spatial / W;
        int xs       = spatial % W;

        // Center (with sub-pixel reg offset)
        float cx = (static_cast<float>(xs) +
                    (reg_data ? reg_data[0 * HW + ys * W + xs] : 0.0f)) * stride_f;
        float cy = (static_cast<float>(ys) +
                    (reg_data ? reg_data[1 * HW + ys * W + xs] : 0.0f)) * stride_f;

        // BBox dimensions
        float bw = (wh_data ? wh_data[0 * HW + ys * W + xs] : 50.0f) * stride_f;
        float bh = (wh_data ? wh_data[1 * HW + ys * W + xs] : 50.0f) * stride_f;
        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

        // Decode K keypoints via helper (hps offsets from center)
        auto lm = decode_peak_keypoints(hps_data, K, K2, HW, W,
                                        ys, xs, cx, cy, stride_f);

        candidates.push_back(CenterPoseResult({x1, y1, x2, y2}, score, cls, std::move(lm)));
    }

    return apply_nms(candidates);
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_nms
// ─────────────────────────────────────────────────────────────────────────────

std::vector<CenterPoseResult> CenterPosePostProcess::apply_nms(
    const std::vector<CenterPoseResult>& detections) const {
    if (detections.empty()) return {};

    // Candidates are already sorted descending (from partial_sort above),
    // but we re-sort here to be safe.
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<CenterPoseResult> results;

    for (size_t idx : indices) {
        if (suppressed[idx]) continue;
        results.push_back(detections[idx]);
        for (size_t jdx : indices) {
            if (suppressed[jdx] || jdx == idx) continue;
            if (detections[idx].iou(detections[jdx]) > nms_threshold_) {
                suppressed[jdx] = true;
            }
        }
    }
    return results;
}
