// Copyright (C) 2018- DEEPX Ltd. All rights reserved.
/**
 * @file retinaface_postprocess.cpp
 * @brief RetinaFace anchor-based face detection with 5-point landmarks
 *
 * Algorithm:
 *   1. Generate multi-scale anchor priors (cached).
 *   2. Auto-detect tensors by last dimension (4=bbox, 2=score, 10=lm).
 *   3. Apply softmax on 2-class logits → face probability.
 *   4. Top-K filtering, then decode bbox deltas and landmark deltas with variance.
 *   5. Apply IoU-NMS.
 *
 * Coordinate output: model-input pixel space [0, input_w] × [0, input_h].
 */

#include "retinaface_postprocess.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <cmath>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

RetinaFacePostProcess::RetinaFacePostProcess(int input_w, int input_h,
                                             float score_threshold,
                                             float nms_threshold)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold) {}

RetinaFacePostProcess::RetinaFacePostProcess()
    : input_width_(640), input_height_(640),
      score_threshold_(0.5f), nms_threshold_(0.4f) {}

// ─────────────────────────────────────────────────────────────────────────────
// Anchor generation (cached, called once per unique input size)
// ─────────────────────────────────────────────────────────────────────────────

void RetinaFacePostProcess::generate_priors() const {
    priors_.clear();

    auto generate_stride_priors = [&](int stride, const std::vector<int>& ms,
                                      int feat_h, int feat_w) {
        for (int i = 0; i < feat_h; ++i) {
            for (int j = 0; j < feat_w; ++j) {
                float cx_n = (j + 0.5f) * stride / static_cast<float>(input_width_);
                float cy_n = (i + 0.5f) * stride / static_cast<float>(input_height_);
                for (int s : ms) {
                    float sw_n = static_cast<float>(s) / static_cast<float>(input_width_);
                    float sh_n = static_cast<float>(s) / static_cast<float>(input_height_);
                    priors_.push_back({cx_n, cy_n, sw_n, sh_n});
                }
            }
        }
    };

    for (int k = 0; k < static_cast<int>(strides_.size()); ++k) {
        int stride = strides_[k];
        int feat_h = (input_height_ + stride - 1) / stride;
        int feat_w = (input_width_  + stride - 1) / stride;
        generate_stride_priors(stride, min_sizes_[k], feat_h, feat_w);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax helper: returns face probability from 2-class logit pair
// ─────────────────────────────────────────────────────────────────────────────

float RetinaFacePostProcess::softmax2_face(float bg, float fg) {
    float m = std::max(bg, fg);
    float e_bg = std::exp(bg - m);
    float e_fg = std::exp(fg - m);
    return e_fg / (e_bg + e_fg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor identification by last dimension
// ─────────────────────────────────────────────────────────────────────────────

RetinaFacePostProcess::IdentifiedTensors_
RetinaFacePostProcess::identifyTensors_(const dxrt::TensorPtrs& outputs) const {
    IdentifiedTensors_ result;
    for (const auto& t : outputs) {
        auto sh = t->shape();
        if (sh.empty()) continue;
        auto last = static_cast<int>(sh.back());
        if (last == 4  && result.bbox     == nullptr) result.bbox     = t.get();
        else if (last == 2  && result.score    == nullptr) result.score    = t.get();
        else if (last == 10 && result.landmark == nullptr) result.landmark = t.get();
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// NHWC feature-map parsing
// ─────────────────────────────────────────────────────────────────────────────

bool RetinaFacePostProcess::isNHWC_(const dxrt::TensorPtrs& outputs) {
    // Flat outputs are 3D ([1, N, k]); NHWC feature maps are 4D ([1, H, W, C]).
    for (const auto& t : outputs) {
        if (t->shape().size() == 4) return true;
    }
    return false;
}

bool RetinaFacePostProcess::parseNHWC_(const dxrt::TensorPtrs& outputs,
                                       std::vector<float>& bbox_out,
                                       std::vector<float>& score_out,
                                       std::vector<float>& lmk_out) const {
    // Group 4D tensors by spatial size (H, W).
    struct Entry { const float* data; int C; };
    std::map<std::pair<int, int>, std::vector<Entry>> groups;
    for (const auto& t : outputs) {
        auto sh = t->shape();
        if (sh.size() != 4) continue;
        int H = static_cast<int>(sh[1]);
        int W = static_cast<int>(sh[2]);
        int C = static_cast<int>(sh[3]);
        groups[{H, W}].push_back({static_cast<const float*>(t->data()), C});
    }
    if (groups.empty()) return false;

    // Process from largest spatial (stride 8) to smallest, matching the stride
    // order and (row, col, anchor) enumeration of generate_priors().
    std::vector<std::pair<int, int>> keys;
    for (const auto& kv : groups) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.first > b.first;
              });

    for (const auto& key : keys) {
        const int H = key.first, W = key.second;
        const auto& entries = groups[key];
        // Channels are A*4 (bbox), A*2 (score), A*10 (landmark). The score map
        // has the smallest channel count → A = minC / 2.
        int minC = entries.front().C;
        for (const auto& e : entries) minC = std::min(minC, e.C);
        const int A = minC / 2;
        if (A <= 0) continue;

        const float* bbox_d = nullptr;
        const float* score_d = nullptr;
        const float* lmk_d = nullptr;
        for (const auto& e : entries) {
            if (e.C == 4 * A) bbox_d = e.data;
            else if (e.C == 2 * A) score_d = e.data;
            else if (e.C == 10 * A) lmk_d = e.data;
        }
        if (!bbox_d || !score_d) continue;

        const int HW = H * W;
        for (int cell = 0; cell < HW; ++cell) {
            for (int a = 0; a < A; ++a) {
                const int base = cell * A + a;  // anchor-major layout within C
                for (int v = 0; v < 4; ++v)  bbox_out.push_back(bbox_d[base * 4 + v]);
                for (int v = 0; v < 2; ++v)  score_out.push_back(score_d[base * 2 + v]);
                if (lmk_d)
                    for (int v = 0; v < 10; ++v) lmk_out.push_back(lmk_d[base * 10 + v]);
            }
        }
    }
    return !bbox_out.empty();
}

// ─────────────────────────────────────────────────────────────────────────────
// postprocess
// ─────────────────────────────────────────────────────────────────────────────

std::vector<RetinaFaceResult> RetinaFacePostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {

    if (outputs.empty()) return {};

    // Lazy-generate priors on first call
    if (priors_.empty()) generate_priors();

    // Two output layouts are supported:
    //   - flat: [1, N, 4] / [1, N, 2] / [1, N, 10]  (last-dim auto-detect)
    //   - NHWC feature maps: [1, H, W, A*4] / [1, H, W, A*2] / [1, H, W, A*10]
    // NHWC is flattened into the same per-prior order as the flat decode below.
    std::vector<float> bbox_buf, score_buf, lmk_buf;
    const float* bbox_data = nullptr;
    const float* score_data = nullptr;
    const float* lmk_data = nullptr;
    int n = 0;

    if (isNHWC_(outputs)) {
        if (!parseNHWC_(outputs, bbox_buf, score_buf, lmk_buf)) return {};
        bbox_data  = bbox_buf.data();
        score_data = score_buf.data();
        lmk_data   = lmk_buf.empty() ? nullptr : lmk_buf.data();
        n = static_cast<int>(score_buf.size() / 2);
    } else {
        if (outputs.size() < 2) return {};
        auto tensors = identifyTensors_(outputs);
        if (tensors.bbox == nullptr || tensors.score == nullptr) return {};
        auto sh_score = tensors.score->shape();
        n = static_cast<int>(sh_score.size() >= 2 ? sh_score[sh_score.size() - 2] : 1);
        bbox_data  = static_cast<const float*>(tensors.bbox->data());
        score_data = static_cast<const float*>(tensors.score->data());
        lmk_data   = (tensors.landmark != nullptr)
                         ? static_cast<const float*>(tensors.landmark->data())
                         : nullptr;
    }

    int n_priors = static_cast<int>(priors_.size());
    n = std::min(n, n_priors);

    const float W = static_cast<float>(input_width_);
    const float H = static_cast<float>(input_height_);

    // Collect candidates
    std::vector<std::pair<float, int>> scored;  // (face_score, index)
    scored.reserve(n);
    for (int i = 0; i < n; ++i) {
        float face_score = softmax2_face(score_data[i * 2], score_data[i * 2 + 1]);
        if (face_score >= score_threshold_) {
            scored.emplace_back(face_score, i);
        }
    }

    // Top-K (by score descending)
    if (static_cast<int>(scored.size()) > top_k_) {
        std::partial_sort(scored.begin(), scored.begin() + top_k_, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        scored.resize(top_k_);
    } else {
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Decode
    std::vector<RetinaFaceResult> candidates;
    candidates.reserve(scored.size());

    for (size_t si = 0; si < scored.size(); ++si) {
        float score = scored[si].first;
        int   idx   = scored[si].second;
        const auto& prior = priors_[idx];
        float pcx = prior[0], pcy = prior[1], psw = prior[2], psh = prior[3];

        // Decode center-form bbox
        float cx = (pcx + bbox_data[idx * 4 + 0] * var0_ * psw) * W;
        float cy = (pcy + bbox_data[idx * 4 + 1] * var0_ * psh) * H;
        float bw = (psw * std::exp(bbox_data[idx * 4 + 2] * var1_)) * W;
        float bh = (psh * std::exp(bbox_data[idx * 4 + 3] * var1_)) * H;

        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

        // Decode landmarks (5 × 2)
        std::vector<float> lm(10, 0.0f);
        if (lmk_data != nullptr) {
            for (int k = 0; k < 5; ++k) {
                lm[k * 2]     = (pcx + lmk_data[idx * 10 + k * 2]     * var0_ * psw) * W;
                lm[k * 2 + 1] = (pcy + lmk_data[idx * 10 + k * 2 + 1] * var0_ * psh) * H;
            }
        }

        candidates.push_back(RetinaFaceResult({x1, y1, x2, y2}, score, std::move(lm)));
    }

    return apply_nms(candidates);
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_nms
// ─────────────────────────────────────────────────────────────────────────────

std::vector<RetinaFaceResult> RetinaFacePostProcess::apply_nms(
    const std::vector<RetinaFaceResult>& detections) const {
    if (detections.empty()) return {};

    // Already sorted by confidence descending from the caller, but sort again to be safe
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<RetinaFaceResult> results;

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
