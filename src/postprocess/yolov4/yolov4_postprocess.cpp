#include "yolov4_postprocess.h"

#include <algorithm>
#include <iostream>
#include <numeric>

YOLOv4PostProcess::YOLOv4PostProcess(
    int input_w, int input_h,
    float score_threshold, float nms_threshold,
    int num_classes, bool normalized)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      num_classes_(num_classes),
      normalized_(normalized) {}

YOLOv4PostProcess::YOLOv4PostProcess()
    : input_width_(512),
      input_height_(512),
      score_threshold_(0.3f),
      nms_threshold_(0.45f),
      num_classes_(80),
      normalized_(true) {}

float YOLOv4Result::iou(const YOLOv4Result& other) const {
    float x1 = std::max(box[0], other.box[0]);
    float y1 = std::max(box[1], other.box[1]);
    float x2 = std::min(box[2], other.box[2]);
    float y2 = std::min(box[3], other.box[3]);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = area() + other.area() - inter;

    return union_area > 0.0f ? inter / union_area : 0.0f;
}

std::vector<YOLOv4Result> YOLOv4PostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.size() < 2) {
        std::cerr << "YOLOv4PostProcess: expected 2 output tensors, got "
                  << outputs.size() << std::endl;
        return {};
    }

    // Auto-detect which tensor holds boxes (last dim == 4) vs scores.
    int box_idx = -1;
    int score_idx = -1;
    for (size_t t = 0; t < outputs.size() && t < 2; ++t) {
        const auto& shp = outputs[t]->shape();
        if (!shp.empty() && static_cast<int>(shp.back()) == 4) {
            box_idx = static_cast<int>(t);
        } else {
            score_idx = static_cast<int>(t);
        }
    }
    if (box_idx < 0 || score_idx < 0) {
        std::cerr << "YOLOv4PostProcess: could not identify boxes/scores tensors"
                  << std::endl;
        return {};
    }

    const float* box_data = static_cast<const float*>(outputs[box_idx]->data());
    const float* score_data = static_cast<const float*>(outputs[score_idx]->data());

    // Number of detections and class columns from the scores tensor.
    const auto& score_shape = outputs[score_idx]->shape();
    int num_boxes = 0;
    int score_cols = 0;
    if (score_shape.size() >= 2) {
        score_cols = static_cast<int>(score_shape.back());
        num_boxes = 1;
        for (size_t d = 0; d + 1 < score_shape.size(); ++d) {
            num_boxes *= static_cast<int>(score_shape[d]);
        }
    } else {
        std::cerr << "YOLOv4PostProcess: unsupported score tensor rank" << std::endl;
        return {};
    }

    const int nc = std::min(score_cols, num_classes_);
    const float scale_x = normalized_ ? static_cast<float>(input_width_) : 1.0f;
    const float scale_y = normalized_ ? static_cast<float>(input_height_) : 1.0f;

    std::vector<YOLOv4Result> candidates;
    for (int i = 0; i < num_boxes; ++i) {
        const float* scores = score_data + static_cast<size_t>(i) * score_cols;

        int best_cls = 0;
        float best_score = scores[0];
        for (int c = 1; c < nc; ++c) {
            if (scores[c] > best_score) {
                best_score = scores[c];
                best_cls = c;
            }
        }

        if (best_score < score_threshold_) continue;

        // Box is x1, y1, x2, y2 (normalized when normalized_ is set).
        const float* b = box_data + static_cast<size_t>(i) * 4;
        candidates.emplace_back(
            std::vector<float>{b[0] * scale_x, b[1] * scale_y,
                               b[2] * scale_x, b[3] * scale_y},
            best_score, best_cls);
    }

    return apply_nms(candidates);
}

std::vector<YOLOv4Result> YOLOv4PostProcess::apply_nms(
    const std::vector<YOLOv4Result>& detections) const {
    if (detections.empty()) return {};

    // Global (cross-class) NMS, matching the golden YOLOv5Postprocessor which
    // applies cv2.dnn.NMSBoxes over all classes at once.
    std::vector<YOLOv4Result> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const YOLOv4Result& a, const YOLOv4Result& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv4Result> result;

    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(sorted_detections[i]);
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (sorted_detections[i].iou(sorted_detections[j]) > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}
