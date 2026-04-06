/**
 * @file obb_postprocessor.hpp
 * @brief Unified OBB (Oriented Bounding Box) Postprocessors for v3 interface
 * 
 * Implements YOLOv26OBBPostprocessor for oriented bounding box detection.
 * 
 * Output tensor format from model: (N, 7) where each row is:
 *   [cx, cy, w, h, score, class_id, angle]
 */

#ifndef OBB_POSTPROCESSOR_HPP
#define OBB_POSTPROCESSOR_HPP

#include <cmath>
#include <algorithm>
#include "common/base/i_processor.hpp"

namespace dxapp {

namespace detail {

/**
 * @brief Scale OBB results from preprocessed coordinates to original coordinates
 */
inline void scaleOBBResults(std::vector<OBBResult>& results,
                            const PreprocessContext& ctx) {
    for (auto& obb : results) {
        obb.cx = (obb.cx - ctx.pad_x) / ctx.scale;
        obb.cy = (obb.cy - ctx.pad_y) / ctx.scale;
        obb.width = obb.width / ctx.scale;
        obb.height = obb.height / ctx.scale;
    }
}

/**
 * @brief Regularize OBB rotation angle
 * 
 * If angle >= pi/2, swap width/height and normalize angle to [0, pi/2)
 */
inline void regularizeOBB(OBBResult& obb) {
    constexpr float PI = 3.14159265358979f;
    constexpr float HALF_PI = PI / 2.0f;

    float t = std::fmod(obb.angle, PI);
    if (t < 0) t += PI;

    if (t >= HALF_PI) {
        std::swap(obb.width, obb.height);
        t = std::fmod(t, HALF_PI);
    }
    obb.angle = t;
}

}  // namespace detail

/**
 * @brief DOTA v1 class labels (15 classes for aerial/satellite object detection)
 */
static const std::vector<std::string> DOTAV1_LABELS = {
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool"
};

/**
 * @brief YOLOv26 OBB postprocessor
 * 
 * Processes model outputs in format (N, 7): [cx, cy, w, h, score, class_id, angle]
 * and converts to OBBResult with coordinate scaling and angle regularization.
 */
class YOLOv26OBBPostprocessor : public IPostprocessor<OBBResult> {
public:
    YOLOv26OBBPostprocessor(int input_width = 640, int input_height = 640,
                            float score_threshold = 0.3f,
                            bool is_ort_configured = false)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold),
          is_ort_configured_(is_ort_configured) {}

    std::vector<OBBResult> process(const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};

        auto& tensor = outputs[0];
        auto shape = tensor->shape();

        // Expected shape: (N, 7) or (1, N, 7)
        int num_detections = 0;
        int cols = 0;

        if (shape.size() == 2) {
            num_detections = static_cast<int>(shape[0]);
            cols = static_cast<int>(shape[1]);
        } else if (shape.size() == 3) {
            num_detections = static_cast<int>(shape[1]);
            cols = static_cast<int>(shape[2]);
        } else {
            return {};
        }

        if (cols < 7) return {};
        const float* data = static_cast<const float*>(tensor->data());

        std::vector<OBBResult> results;
        results.reserve(num_detections);

        for (int i = 0; i < num_detections; ++i) {
            const float* row = data + i * cols;
            float score = row[4];

            if (score < score_threshold_) continue;

            OBBResult obb;
            obb.cx = row[0];
            obb.cy = row[1];
            obb.width = row[2];
            obb.height = row[3];
            obb.confidence = score;
            obb.class_id = static_cast<int>(row[5]);
            obb.angle = row[6];

            // Assign class name from DOTA labels
            if (obb.class_id >= 0 && obb.class_id < static_cast<int>(DOTAV1_LABELS.size())) {
                obb.class_name = DOTAV1_LABELS[obb.class_id];
            } else {
                obb.class_name = "class_" + std::to_string(obb.class_id);
            }

            results.push_back(std::move(obb));
        }

        // Scale coordinates back to original image space
        detail::scaleOBBResults(results, ctx);

        // Regularize angles
        for (auto& obb : results) {
            detail::regularizeOBB(obb);
        }

        return results;
    }

    std::string getModelName() const override { return "YOLOv26OBB"; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    bool is_ort_configured_;
};

}  // namespace dxapp

#endif  // OBB_POSTPROCESSOR_HPP
