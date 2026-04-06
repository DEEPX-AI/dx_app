#include "yolov8pose_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>

YOLOv8PosePostProcess::YOLOv8PosePostProcess(
    int input_w, int input_h, float score_threshold, float nms_threshold,
    bool is_ort_configured)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      is_ort_configured_(is_ort_configured) {}

YOLOv8PosePostProcess::YOLOv8PosePostProcess()
    : input_width_(640),
      input_height_(640),
      score_threshold_(0.3f),
      nms_threshold_(0.45f),
      is_ort_configured_(false) {}

float YOLOv8PoseResult::iou(const YOLOv8PoseResult& other) const {
    float x1 = std::max(box[0], other.box[0]);
    float y1 = std::max(box[1], other.box[1]);
    float x2 = std::min(box[2], other.box[2]);
    float y2 = std::min(box[3], other.box[3]);

    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = area() + other.area() - inter_area;

    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

std::vector<YOLOv8PoseResult> YOLOv8PosePostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    auto detections = decode_outputs(outputs);
    return apply_nms(detections);
}

std::vector<YOLOv8PoseResult> YOLOv8PosePostProcess::decode_outputs(
    const dxrt::TensorPtrs& outputs) const {
    if (outputs.empty()) return {};

    const auto& shape = outputs[0]->shape();
    const float* data = static_cast<const float*>(outputs[0]->data());

    // Expected shape: [1, 56, N] where 56 = 4 bbox + 1 score + 51 keypoints (17*3)
    // Or after ORT: [1, N, 56]
    int num_features, num_anchors;
    bool transposed = false;

    if (shape.size() == 3) {
        int dim1 = static_cast<int>(shape[1]);
        int dim2 = static_cast<int>(shape[2]);

        if (dim1 == 56 || dim1 == (4 + 1 + NUM_KEYPOINTS * 3)) {
            // [1, 56, N] — standard
            num_features = dim1;
            num_anchors = dim2;
            transposed = false;
        } else {
            // [1, N, 56] — transposed (ORT style)
            num_anchors = dim1;
            num_features = dim2;
            transposed = true;
        }
    } else if (shape.size() == 2) {
        // [N, 56]
        num_anchors = static_cast<int>(shape[0]);
        num_features = static_cast<int>(shape[1]);
        transposed = true;
    } else {
        std::cerr << "YOLOv8PosePostProcess: unsupported tensor rank " << shape.size() << std::endl;
        return {};
    }

    const int kps_count = NUM_KEYPOINTS * 3;  // 51
    if (num_features < 5 + kps_count) {
        std::cerr << "YOLOv8PosePostProcess: feature count " << num_features
                  << " too small for pose (need " << (5 + kps_count) << ")" << std::endl;
        return {};
    }

    std::vector<YOLOv8PoseResult> results;

    for (int i = 0; i < num_anchors; ++i) {
        // Access elements based on layout
        auto get_val = [&](int feature_idx) -> float {
            if (transposed) {
                return data[i * num_features + feature_idx];
            } else {
                return data[feature_idx * num_anchors + i];
            }
        };

        // Score
        float score = get_val(4);

        if (score < score_threshold_) continue;

        // Bbox: cx, cy, w, h → x1, y1, x2, y2
        float cx = get_val(0);
        float cy = get_val(1);
        float w = get_val(2);
        float h = get_val(3);

        std::vector<float> box = {
            cx - w * 0.5f,
            cy - h * 0.5f,
            cx + w * 0.5f,
            cy + h * 0.5f
        };

        // Keypoints: 17 * (x, y, conf)
        std::vector<float> landmarks(kps_count);
        for (int j = 0; j < kps_count; ++j) {
            landmarks[j] = get_val(5 + j);
        }

        results.emplace_back(std::move(box), score, std::move(landmarks));
    }

    return results;
}

std::vector<YOLOv8PoseResult> YOLOv8PosePostProcess::apply_nms(
    const std::vector<YOLOv8PoseResult>& detections) const {
    if (detections.empty()) return {};

    // Sort by confidence (descending)
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&detections](size_t a, size_t b) {
                  return detections[a].confidence > detections[b].confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<YOLOv8PoseResult> results;

    for (size_t idx : indices) {
        if (suppressed[idx]) continue;

        results.push_back(detections[idx]);

        for (size_t j_idx : indices) {
            if (suppressed[j_idx] || j_idx == idx) continue;
            if (detections[idx].iou(detections[j_idx]) > nms_threshold_) {
                suppressed[j_idx] = true;
            }
        }
    }

    return results;
}
