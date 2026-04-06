#include "ssd_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

SSDPostProcess::SSDPostProcess(int input_w, int input_h,
                               float score_threshold, float nms_threshold,
                               int num_classes, bool has_background)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      num_classes_(num_classes),
      has_background_(has_background) {}

SSDPostProcess::SSDPostProcess()
    : input_width_(300),
      input_height_(300),
      score_threshold_(0.3f),
      nms_threshold_(0.45f),
      num_classes_(20),
      has_background_(true) {}

float SSDResult::iou(const SSDResult& other) const {
    float x1 = std::max(box[0], other.box[0]);
    float y1 = std::max(box[1], other.box[1]);
    float x2 = std::min(box[2], other.box[2]);
    float y2 = std::min(box[3], other.box[3]);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = area() + other.area() - inter;

    return union_area > 0.0f ? inter / union_area : 0.0f;
}

std::vector<SSDResult> SSDPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.size() < 2) {
        std::cerr << "SSDPostProcess: expected 2 output tensors, got " << outputs.size() << std::endl;
        return {};
    }

    // output[0]: scores [1, N, C+1] or [N, C+1]
    // output[1]: boxes  [1, N, 4]  or [N, 4]
    const auto& score_shape = outputs[0]->shape();
    const auto& box_shape = outputs[1]->shape();
    const float* score_data = static_cast<const float*>(outputs[0]->data());
    const float* box_data = static_cast<const float*>(outputs[1]->data());

    int num_boxes;
    int score_cols;
    if (score_shape.size() == 3) {
        num_boxes = static_cast<int>(score_shape[1]);
        score_cols = static_cast<int>(score_shape[2]);
    } else if (score_shape.size() == 2) {
        num_boxes = static_cast<int>(score_shape[0]);
        score_cols = static_cast<int>(score_shape[1]);
    } else {
        std::cerr << "SSDPostProcess: unsupported score tensor rank " << score_shape.size() << std::endl;
        return {};
    }

    int fg_offset = has_background_ ? 1 : 0;
    int fg_classes = score_cols - fg_offset;

    // Auto-detect box format: [ymin,xmin,ymax,xmax] vs [x1,y1,x2,y2].
    // DXNN runtime typically outputs [x1,y1,x2,y2] directly.
    // Check which interpretation yields more boxes with height > width
    // (indicates vertical objects like persons — more common).
    // Default to [x1,y1,x2,y2] (no swap) when indeterminate.
    bool swap_xy = false;
    {
        // Collect high-confidence box stats for both interpretations
        int sensible_a = 0;  // [ymin,xmin,ymax,xmax] interpretation
        int sensible_b = 0;  // [x1,y1,x2,y2] interpretation
        int sample_count = 0;
        for (int i = 0; i < num_boxes && sample_count < 200; ++i) {
            const float* scores = score_data + i * score_cols;
            float best_score = 0.0f;
            for (int c = 0; c < fg_classes; ++c) {
                if (scores[fg_offset + c] > best_score)
                    best_score = scores[fg_offset + c];
            }
            if (best_score < score_threshold_) continue;
            sample_count++;

            const float* b = box_data + i * 4;
            // A: [ymin,xmin,ymax,xmax] → w = b[3]-b[1], h = b[2]-b[0]
            float w_a = b[3] - b[1], h_a = b[2] - b[0];
            // B: [x1,y1,x2,y2] → w = b[2]-b[0], h = b[3]-b[1]
            float w_b = b[2] - b[0], h_b = b[3] - b[1];
            if (w_a > 0 && h_a > 0) sensible_a++;
            if (w_b > 0 && h_b > 0) sensible_b++;
        }
        // Only swap when A is strictly better than B
        swap_xy = (sensible_a > sensible_b);
    }

    std::vector<SSDResult> candidates;

    for (int i = 0; i < num_boxes; ++i) {
        const float* scores = score_data + i * score_cols;

        // Find max foreground class
        int best_cls = 0;
        float best_score = scores[fg_offset];
        for (int c = 1; c < fg_classes; ++c) {
            if (scores[fg_offset + c] > best_score) {
                best_score = scores[fg_offset + c];
                best_cls = c;
            }
        }

        if (best_score < score_threshold_) continue;

        // Decode box
        const float* b = box_data + i * 4;

        float x1, y1, x2, y2;
        // Check if normalized (values mostly in [0, ~1.5])
        if (std::fabs(b[0]) < 5.0f && std::fabs(b[1]) < 5.0f &&
            std::fabs(b[2]) < 5.0f && std::fabs(b[3]) < 5.0f) {
            if (swap_xy) {
                // [ymin, xmin, ymax, xmax] → [x1, y1, x2, y2]
                x1 = b[1];
                y1 = b[0];
                x2 = b[3];
                y2 = b[2];
            } else {
                // Already [x1, y1, x2, y2]
                x1 = b[0];
                y1 = b[1];
                x2 = b[2];
                y2 = b[3];
            }
        } else {
            // Input is in pixel coordinates — normalize by input size so that
            // SSDResult.box always contains normalized coordinates.
            x1 = b[0] / static_cast<float>(input_width_);
            y1 = b[1] / static_cast<float>(input_height_);
            x2 = b[2] / static_cast<float>(input_width_);
            y2 = b[3] / static_cast<float>(input_height_);
        }

        candidates.emplace_back(std::vector<float>{x1, y1, x2, y2}, best_score, best_cls);
    }

    return apply_nms(candidates);
}

std::vector<SSDResult> SSDPostProcess::apply_nms(
    const std::vector<SSDResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&detections](size_t a, size_t b) {
                  return detections[a].confidence > detections[b].confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<SSDResult> results;

    for (size_t idx : indices) {
        if (suppressed[idx]) continue;
        results.push_back(detections[idx]);
        for (size_t j : indices) {
            if (suppressed[j] || j == idx) continue;
            if (detections[idx].class_id == detections[j].class_id &&
                detections[idx].iou(detections[j]) > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    return results;
}
