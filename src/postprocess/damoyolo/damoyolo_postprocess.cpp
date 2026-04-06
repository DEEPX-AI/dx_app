#include "damoyolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

DamoYOLOPostProcess::DamoYOLOPostProcess(
    int input_w, int input_h,
    float score_threshold, float nms_threshold, int num_classes)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      num_classes_(num_classes) {}

DamoYOLOPostProcess::DamoYOLOPostProcess()
    : input_width_(640),
      input_height_(640),
      score_threshold_(0.3f),
      nms_threshold_(0.45f),
      num_classes_(80) {}

float DamoYOLOResult::iou(const DamoYOLOResult& other) const {
    float x1 = std::max(box[0], other.box[0]);
    float y1 = std::max(box[1], other.box[1]);
    float x2 = std::min(box[2], other.box[2]);
    float y2 = std::min(box[3], other.box[3]);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = area() + other.area() - inter;

    return union_area > 0.0f ? inter / union_area : 0.0f;
}

std::vector<DamoYOLOResult> DamoYOLOPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.size() < 2) {
        std::cerr << "DamoYOLOPostProcess: expected 2 output tensors, got "
                  << outputs.size() << std::endl;
        return {};
    }

    // output[0]: scores [1, N, C] or [N, C]
    // output[1]: boxes  [1, N, 4] or [N, 4]
    const auto& score_shape = outputs[0]->shape();
    const auto& box_shape = outputs[1]->shape();
    const float* score_data = static_cast<const float*>(outputs[0]->data());
    const float* box_data = static_cast<const float*>(outputs[1]->data());

    int num_boxes, score_cols;
    if (score_shape.size() == 3) {
        num_boxes = static_cast<int>(score_shape[1]);
        score_cols = static_cast<int>(score_shape[2]);
    } else if (score_shape.size() == 2) {
        num_boxes = static_cast<int>(score_shape[0]);
        score_cols = static_cast<int>(score_shape[1]);
    } else {
        std::cerr << "DamoYOLOPostProcess: unsupported score tensor rank" << std::endl;
        return {};
    }

    int nc = std::min(score_cols, num_classes_);
    std::vector<DamoYOLOResult> candidates;

    for (int i = 0; i < num_boxes; ++i) {
        const float* scores = score_data + i * score_cols;

        // Find max class score (already sigmoid'd)
        int best_cls = 0;
        float best_score = scores[0];
        for (int c = 1; c < nc; ++c) {
            if (scores[c] > best_score) {
                best_score = scores[c];
                best_cls = c;
            }
        }

        if (best_score < score_threshold_) continue;

        // Box: already x1y1x2y2 in pixel space
        const float* b = box_data + i * 4;
        candidates.emplace_back(
            std::vector<float>{b[0], b[1], b[2], b[3]}, best_score, best_cls);
    }

    return apply_nms(candidates);
}

std::vector<DamoYOLOResult> DamoYOLOPostProcess::apply_nms(
    const std::vector<DamoYOLOResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&detections](size_t a, size_t b) {
                  return detections[a].confidence > detections[b].confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<DamoYOLOResult> results;

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
