// Copyright (C) 2018- DEEPX Ltd. All rights reserved.
/**
 * @file ulfgfd_postprocess.cpp
 * @brief ULFGFD (Ultra-Light-Fast-Generic Face Detector) postprocessor
 *
 * SSD-style face detection without landmarks.
 * Two output tensors: scores [N,2], boxes [N,4] (normalized).
 */

#include "ulfgfd_postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

ULFGFDPostProcess::ULFGFDPostProcess(int input_w, int input_h,
                                     float score_threshold,
                                     float nms_threshold)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold) {}

ULFGFDPostProcess::ULFGFDPostProcess()
    : input_width_(320), input_height_(240),
      score_threshold_(0.7f), nms_threshold_(0.3f) {}

// ─────────────────────────────────────────────────────────────────────────────
// postprocess
// ─────────────────────────────────────────────────────────────────────────────

std::vector<ULFGFDResult> ULFGFDPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.size() < 2) {
        return {};
    }

    // Identify scores tensor (last dim = 2) and boxes tensor (last dim = 4)
    // by inspecting shape[last].
    dxrt::Tensor* scores_t = nullptr;
    dxrt::Tensor* boxes_t  = nullptr;

    for (const auto& t : outputs) {
        auto sh = t->shape();
        if (sh.empty()) continue;
        int last_dim = static_cast<int>(sh.back());
        if (last_dim == 2 && scores_t == nullptr) {
            scores_t = t.get();
        } else if (last_dim == 4 && boxes_t == nullptr) {
            boxes_t = t.get();
        }
    }

    // Fallback: assign by index order
    if (scores_t == nullptr) scores_t = outputs[0].get();
    if (boxes_t  == nullptr) boxes_t  = outputs[1].get();

    auto score_shape = scores_t->shape();
    auto box_shape   = boxes_t->shape();

    int n_scores = static_cast<int>(score_shape.size() >= 2 ? score_shape[score_shape.size() - 2] : 1);
    int n_boxes  = static_cast<int>(box_shape.size() >= 2  ? box_shape[box_shape.size() - 2]  : 1);
    int n = std::min(n_scores, n_boxes);

    const float* score_ptr = static_cast<const float*>(scores_t->data());
    const float* box_ptr   = static_cast<const float*>(boxes_t->data());

    // Detect box format:
    // 1. Normalized [x1,y1,x2,y2] in [0,1] → max_val <= 2.0, few negatives
    // 2. Pixel [x1,y1,x2,y2] → max_val > 2.0
    // 3. SSD deltas [dcx,dcy,dw,dh] → many negative values (anchor offsets)
    float max_box_val = 0.0f;
    int neg_count = 0;
    int sample_count = std::min(n * 4, 200);
    for (int i = 0; i < sample_count; ++i) {
        max_box_val = std::max(max_box_val, std::abs(box_ptr[i]));
        if (box_ptr[i] < -0.01f) neg_count++;
    }
    bool is_ssd_delta = (neg_count > sample_count / 10);  // >10% negative → likely deltas
    bool normalized = (!is_ssd_delta && max_box_val <= 2.0f);

    // Generate SSD priors if needed (ULFG 320×240 config)
    struct Prior { float cx, cy, w, h; };
    std::vector<Prior> priors;
    if (is_ssd_delta) {
        int min_sizes_list[][3] = {{10,16,24}, {32,48,0}, {64,96,0}, {128,192,256}};
        int min_sizes_count[] = {3, 2, 2, 3};
        int strides[] = {8, 16, 32, 64};
        float iw = static_cast<float>(input_width_);
        float ih = static_cast<float>(input_height_);
        for (int k = 0; k < 4; ++k) {
            int fh = static_cast<int>(std::ceil(ih / strides[k]));
            int fw = static_cast<int>(std::ceil(iw / strides[k]));
            for (int i = 0; i < fh; ++i) {
                for (int j = 0; j < fw; ++j) {
                    for (int s = 0; s < min_sizes_count[k]; ++s) {
                        float ms = static_cast<float>(min_sizes_list[k][s]);
                        priors.push_back({
                            (j + 0.5f) * strides[k] / iw,
                            (i + 0.5f) * strides[k] / ih,
                            ms / iw,
                            ms / ih
                        });
                    }
                }
            }
        }
    }

    std::vector<ULFGFDResult> candidates;
    candidates.reserve(top_k_);

    for (int i = 0; i < n; ++i) {
        // scores_t shape: [..., N, 2] → face score is column 1
        float face_score = score_ptr[i * 2 + 1];
        if (face_score < score_threshold_) continue;

        float bx1, by1, bx2, by2;
        if (is_ssd_delta && i < static_cast<int>(priors.size())) {
            // SSD delta decode with priors
            float variance0 = 0.1f, variance1 = 0.2f;
            float dcx = box_ptr[i * 4 + 0];
            float dcy = box_ptr[i * 4 + 1];
            float dw  = box_ptr[i * 4 + 2];
            float dh  = box_ptr[i * 4 + 3];
            float cx = priors[i].cx + dcx * variance0 * priors[i].w;
            float cy = priors[i].cy + dcy * variance0 * priors[i].h;
            float pw = priors[i].w * std::exp(dw * variance1);
            float ph = priors[i].h * std::exp(dh * variance1);
            bx1 = (cx - pw * 0.5f) * static_cast<float>(input_width_);
            by1 = (cy - ph * 0.5f) * static_cast<float>(input_height_);
            bx2 = (cx + pw * 0.5f) * static_cast<float>(input_width_);
            by2 = (cy + ph * 0.5f) * static_cast<float>(input_height_);
        } else if (normalized) {
            bx1 = box_ptr[i * 4 + 0] * static_cast<float>(input_width_);
            by1 = box_ptr[i * 4 + 1] * static_cast<float>(input_height_);
            bx2 = box_ptr[i * 4 + 2] * static_cast<float>(input_width_);
            by2 = box_ptr[i * 4 + 3] * static_cast<float>(input_height_);
        } else {
            bx1 = box_ptr[i * 4 + 0];
            by1 = box_ptr[i * 4 + 1];
            bx2 = box_ptr[i * 4 + 2];
            by2 = box_ptr[i * 4 + 3];
        }

        if (bx2 <= bx1 || by2 <= by1) continue;

        candidates.push_back(ULFGFDResult({bx1, by1, bx2, by2}, face_score));

        if (static_cast<int>(candidates.size()) >= top_k_) break;
    }

    return apply_nms(candidates);
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_nms
// ─────────────────────────────────────────────────────────────────────────────

std::vector<ULFGFDResult> ULFGFDPostProcess::apply_nms(
    const std::vector<ULFGFDResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<ULFGFDResult> results;

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
