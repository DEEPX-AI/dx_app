#include "obb_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

OBBPostProcess::OBBPostProcess(int input_w, int input_h, float score_threshold)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold) {}

OBBPostProcess::OBBPostProcess()
    : input_width_(640),
      input_height_(640),
      score_threshold_(0.3f) {}

void OBBPostProcess::regularize_angle(float& w, float& h, float& angle) const {
    // Normalize angle to [0, pi)
    angle = std::fmod(angle, static_cast<float>(M_PI));
    if (angle < 0.0f) angle += static_cast<float>(M_PI);

    // If angle >= pi/2, swap w/h and subtract pi/2
    if (angle >= static_cast<float>(M_PI) / 2.0f) {
        std::swap(w, h);
        angle = std::fmod(angle, static_cast<float>(M_PI) / 2.0f);
    }
}

std::vector<OBBResult> OBBPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) return {};

    const auto& shape = outputs[0]->shape();
    const float* data = static_cast<const float*>(outputs[0]->data());

    // Expected: [N, 7] or [1, N, 7]
    int num_detections, num_cols;
    if (shape.size() == 3) {
        num_detections = static_cast<int>(shape[1]);
        num_cols = static_cast<int>(shape[2]);
    } else if (shape.size() == 2) {
        num_detections = static_cast<int>(shape[0]);
        num_cols = static_cast<int>(shape[1]);
    } else {
        std::cerr << "OBBPostProcess: unsupported tensor rank " << shape.size() << std::endl;
        return {};
    }

    if (num_cols < 7) {
        std::cerr << "OBBPostProcess: expected 7 columns, got " << num_cols << std::endl;
        return {};
    }

    std::vector<OBBResult> results;

    for (int i = 0; i < num_detections; ++i) {
        const float* row = data + i * num_cols;

        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        float score = row[4];
        int class_id = static_cast<int>(row[5]);
        float angle = row[6];

        if (score < score_threshold_) continue;

        // Regularize angle
        regularize_angle(w, h, angle);

        results.emplace_back(cx, cy, w, h, angle, score, class_id);
    }

    return results;
}
