#include "nanodet_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

NanoDetPostProcess::NanoDetPostProcess(
    int input_w, int input_h,
    float score_threshold, float nms_threshold,
    int num_classes, int reg_max)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      num_classes_(num_classes),
      reg_max_(reg_max) {
    build_anchors();
}

NanoDetPostProcess::NanoDetPostProcess()
    : input_width_(416),
      input_height_(416),
      score_threshold_(0.3f),
      nms_threshold_(0.45f),
      num_classes_(80),
      reg_max_(10) {
    build_anchors();
}

void NanoDetPostProcess::build_anchors() {
    anchor_cx_.clear();
    anchor_cy_.clear();
    anchor_stride_.clear();

    for (int stride : strides_) {
        int h = input_height_ / stride;
        int w = input_width_ / stride;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                anchor_cx_.push_back((x + 0.5f) * stride);
                anchor_cy_.push_back((y + 0.5f) * stride);
                anchor_stride_.push_back(static_cast<float>(stride));
            }
        }
    }
    total_anchors_ = static_cast<int>(anchor_cx_.size());
}

std::vector<float> NanoDetPostProcess::dfl_decode(const float* reg, int bins) const {
    // Decode 4 sides from DFL distribution
    // reg: [4 * bins] values
    std::vector<float> distances(4);

    for (int side = 0; side < 4; ++side) {
        const float* side_reg = reg + side * bins;

        // Softmax
        float max_val = side_reg[0];
        for (int b = 1; b < bins; ++b) {
            if (side_reg[b] > max_val) max_val = side_reg[b];
        }

        float sum = 0.0f;
        std::vector<float> softmax_vals(bins);
        for (int b = 0; b < bins; ++b) {
            softmax_vals[b] = std::exp(side_reg[b] - max_val);
            sum += softmax_vals[b];
        }

        // Weighted sum: distance = sum(softmax[i] * i)
        float dist = 0.0f;
        if (sum > 0.0f) {
            for (int b = 0; b < bins; ++b) {
                dist += (softmax_vals[b] / sum) * b;
            }
        }
        distances[side] = dist;
    }

    return distances;
}

float NanoDetResult::iou(const NanoDetResult& other) const {
    float x1 = std::max(box[0], other.box[0]);
    float y1 = std::max(box[1], other.box[1]);
    float x2 = std::min(box[2], other.box[2]);
    float y2 = std::min(box[3], other.box[3]);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = area() + other.area() - inter;

    return union_area > 0.0f ? inter / union_area : 0.0f;
}

std::vector<NanoDetResult> NanoDetPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) return {};

    const auto& shape = outputs[0]->shape();
    const float* data = static_cast<const float*>(outputs[0]->data());

    // Expected: [1, N, C + 4*(reg_max+1)] or [N, C + 4*(reg_max+1)]
    int num_anchors, num_cols;
    if (shape.size() == 3) {
        num_anchors = static_cast<int>(shape[1]);
        num_cols = static_cast<int>(shape[2]);
    } else if (shape.size() == 2) {
        num_anchors = static_cast<int>(shape[0]);
        num_cols = static_cast<int>(shape[1]);
    } else {
        std::cerr << "NanoDetPostProcess: unsupported tensor rank " << shape.size() << std::endl;
        return {};
    }

    int bins = reg_max_ + 1;
    int expected_cols = num_classes_ + 4 * bins;

    if (num_cols != expected_cols) {
        std::cerr << "NanoDetPostProcess: expected " << expected_cols
                  << " columns but got " << num_cols << std::endl;
        return {};
    }

    // Use min of actual anchors and predicted anchors
    int n = std::min(num_anchors, total_anchors_);

    // Auto-detect whether class scores are already post-sigmoid (values in [0,1])
    // or raw logits (can be negative). Sample the first few rows.
    bool already_sigmoid = true;
    {
        int sample_n = std::min(n, 200);
        for (int i = 0; i < sample_n && already_sigmoid; ++i) {
            const float* row = data + i * num_cols;
            for (int c = 0; c < num_classes_; ++c) {
                if (row[c] < 0.0f || row[c] > 1.01f) {
                    already_sigmoid = false;
                    break;
                }
            }
        }
    }

    std::vector<NanoDetResult> candidates;

    for (int i = 0; i < n; ++i) {
        const float* row = data + i * num_cols;

        // Find max class score
        int best_cls = 0;
        float best_score = already_sigmoid ? row[0] : sigmoid(row[0]);
        for (int c = 1; c < num_classes_; ++c) {
            float s = already_sigmoid ? row[c] : sigmoid(row[c]);
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
        }

        if (best_score < score_threshold_) continue;

        // DFL decode → [left, top, right, bottom] in stride units
        std::vector<float> distances = dfl_decode(row + num_classes_, bins);

        // Scale by stride
        float stride = anchor_stride_[i];
        float left = distances[0] * stride;
        float top = distances[1] * stride;
        float right = distances[2] * stride;
        float bottom = distances[3] * stride;

        // Convert to x1y1x2y2
        float x1 = anchor_cx_[i] - left;
        float y1 = anchor_cy_[i] - top;
        float x2 = anchor_cx_[i] + right;
        float y2 = anchor_cy_[i] + bottom;

        candidates.emplace_back(
            std::vector<float>{x1, y1, x2, y2}, best_score, best_cls);
    }

    return apply_nms(candidates);
}

std::vector<NanoDetResult> NanoDetPostProcess::apply_nms(
    const std::vector<NanoDetResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&detections](size_t a, size_t b) {
                  return detections[a].confidence > detections[b].confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<NanoDetResult> results;

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
