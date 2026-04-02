/**
 * @file anchor_face_v7_postprocessor.hpp
 * @brief YOLOv7-Face Post-processing
 *
 * Part of DX-APP v3.0.0 refactoring.
 *
 * YOLOv7-Face models output a pre-decoded tensor [1, N, 21] alongside
 * per-scale NPU tensors [1, 3, H, W, 21].
 *
 * Format per detection (21 values):
 *   [x, y, w, h, obj_conf,
 *    lm0_x, lm0_y, lm0_conf,
 *    lm1_x, lm1_y, lm1_conf,
 *    lm2_x, lm2_y, lm2_conf,
 *    lm3_x, lm3_y, lm3_conf,
 *    lm4_x, lm4_y, lm4_conf,
 *    cls_conf]
 *
 * This postprocessor uses the pre-decoded [1, N, 21] tensor directly,
 * avoiding anchor-based grid decoding entirely.
 *
 * Key differences from YOLOv5Face (16 values):
 *   - YOLOv5Face landmarks: 5 points × 2 coords = 10 values
 *   - YOLOv7Face landmarks: 5 points × 3 (x, y, conf) = 15 values
 *   - YOLOv7 W6 models: 4 scales (stride 8,16,32,64) vs YOLOv5's 3 scales
 *   - YOLOv7 NPU tensors: 5D [1,3,H,W,21] vs YOLOv5's 4D [1,48,H,W]
 */

#ifndef ANCHOR_FACE_V7_POSTPROCESSOR_HPP
#define ANCHOR_FACE_V7_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Re-use YOLOv5FaceResult structure (box + confidence + landmarks xy)
#include "anchor_face_postprocessor.hpp"

/**
 * @brief YOLOv7-Face post-processing class
 *
 * Handles the pre-decoded [1, N, 21] output tensor from YOLOv7-Face models.
 * Also supports 5D per-scale NPU tensors [1, 3, H, W, 21] with anchor decoding.
 */
class YOLOv7FacePostProcess {
public:
    YOLOv7FacePostProcess(int input_w = 640, int input_h = 640,
                          float obj_threshold = 0.5f,
                          float score_threshold = 0.5f,
                          float nms_threshold = 0.45f,
                          bool is_ort_configured = false)
        : input_width_(input_w),
          input_height_(input_h),
          object_threshold_(obj_threshold),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold),
          is_ort_configured_(is_ort_configured) {}

    ~YOLOv7FacePostProcess() = default;

    /**
     * @brief Main postprocess entry point
     *
     * Strategy:
     *   1. Look for a 3D pre-decoded tensor [1, N, 21] → use fast path
     *   2. Else fall back to 5D per-scale NPU decoding
     */
    std::vector<YOLOv5FaceResult> postprocess(const dxrt::TensorPtrs& outputs) {
        std::vector<YOLOv5FaceResult> detections;

        // Strategy 1: Find the pre-decoded 3D tensor [1, N, 21]
        const dxrt::TensorPtr* decoded_tensor = nullptr;
        for (const auto& output : outputs) {
            if (output->shape().size() == 3 && output->shape()[2] == kValuesPerDet) {
                decoded_tensor = &output;
                break;
            }
        }

        if (decoded_tensor) {
            detections = decode_predecoded(*decoded_tensor);
        } else {
            // Strategy 2: 5D per-scale NPU tensors [1, 3, H, W, 21]
            detections = decode_npu_5d(outputs);
        }

        // Apply NMS
        return apply_nms(detections);
    }

    // Accessors
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }

private:
    static constexpr int kValuesPerDet = 21;      // YOLOv7Face: 21 values per detection
    static constexpr int kNumLandmarks = 5;        // 5 facial landmark points
    static constexpr int kLandmarkStride = 3;      // x, y, conf per landmark
    static constexpr int kBoxOffset = 0;           // box starts at index 0
    static constexpr int kObjConfOffset = 4;       // obj_conf at index 4
    static constexpr int kLandmarkOffset = 5;      // landmarks start at index 5
    static constexpr int kClsConfOffset = 20;      // cls_conf at index 20

    int input_width_;
    int input_height_;
    float object_threshold_;
    float score_threshold_;
    float nms_threshold_;
    bool is_ort_configured_;

    /**
     * @brief Decode pre-decoded [1, N, 21] tensor (fast path)
     *
     * This is the preferred path. The DXNN model already outputs
     * decoded detections in [x, y, w, h, ...] center format.
     */
    std::vector<YOLOv5FaceResult> decode_predecoded(const dxrt::TensorPtr& tensor) const {
        std::vector<YOLOv5FaceResult> detections;
        const float* data = static_cast<const float*>(tensor->data());
        const int num_dets = static_cast<int>(tensor->shape()[1]);

        detections.reserve(256);  // Pre-allocate for typical face count

        for (int i = 0; i < num_dets; ++i) {
            const float* det = data + i * kValuesPerDet;

            // obj_conf filter
            float obj_conf = det[kObjConfOffset];
            if (obj_conf < object_threshold_) continue;

            // cls_conf × obj_conf filter
            float cls_conf = det[kClsConfOffset];
            float conf = obj_conf * cls_conf;
            if (conf < score_threshold_) continue;

            YOLOv5FaceResult result;
            result.confidence = conf;

            // Box: center (x,y,w,h) → corner (x1,y1,x2,y2)
            float cx = det[0], cy = det[1], w = det[2], h = det[3];
            result.box = {cx - w / 2.0f, cy - h / 2.0f,
                          cx + w / 2.0f, cy + h / 2.0f};

            // Landmarks: extract x, y (skip per-landmark confidence)
            result.landmarks.reserve(kNumLandmarks * 2);
            for (int k = 0; k < kNumLandmarks; ++k) {
                int base = kLandmarkOffset + k * kLandmarkStride;
                result.landmarks.push_back(det[base + 0]);  // x
                result.landmarks.push_back(det[base + 1]);  // y
                // det[base + 2] = landmark confidence (ignored for compatibility)
            }

            detections.push_back(std::move(result));
        }
        return detections;
    }

    /**
     * @brief Decode 5D per-scale NPU tensors [1, 3, H, W, 21]
     *
     * Fallback for when no pre-decoded tensor is available.
     * Supports variable number of scales (3 or 4).
     */
    std::vector<YOLOv5FaceResult> decode_npu_5d(const dxrt::TensorPtrs& outputs) const {
        std::vector<YOLOv5FaceResult> detections;

        // Collect 5D tensors, sorted by grid size (largest first = smallest stride)
        std::vector<const dxrt::TensorPtr*> scale_tensors;
        for (const auto& output : outputs) {
            if (output->shape().size() == 5 && output->shape()[4] == kValuesPerDet) {
                scale_tensors.push_back(&output);
            }
        }

        if (scale_tensors.empty()) {
            throw std::runtime_error("[DXAPP] YOLOv7Face: No valid 5D tensors found");
        }

        // Sort by grid height descending (stride 8 first)
        std::sort(scale_tensors.begin(), scale_tensors.end(),
                  [](const dxrt::TensorPtr* a, const dxrt::TensorPtr* b) {
                      return (*a)->shape()[2] > (*b)->shape()[2];
                  });

        for (const auto* tensor_ptr : scale_tensors) {
            const auto& tensor = *tensor_ptr;
            const float* data = static_cast<const float*>(tensor->data());
            const int num_anchors = static_cast<int>(tensor->shape()[1]);  // 3
            const int grid_h = static_cast<int>(tensor->shape()[2]);
            const int grid_w = static_cast<int>(tensor->shape()[3]);
            const int stride = input_height_ / grid_h;

            for (int a = 0; a < num_anchors; ++a) {
                for (int gy = 0; gy < grid_h; ++gy) {
                    for (int gx = 0; gx < grid_w; ++gx) {
                        // Index into [1, 3, H, W, 21] tensor (NHWC-like last dim)
                        int base_idx = ((a * grid_h + gy) * grid_w + gx) * kValuesPerDet;
                        const float* det = data + base_idx;

                        float obj_conf = sigmoid(det[kObjConfOffset]);
                        float cls_conf = sigmoid(det[kClsConfOffset]);
                        float conf = obj_conf * cls_conf;
                        if (conf < score_threshold_) continue;

                        YOLOv5FaceResult result;
                        result.confidence = conf;

                        // Box decode with grid offset
                        float cx = (sigmoid(det[0]) * 2.0f - 0.5f + gx) * stride;
                        float cy = (sigmoid(det[1]) * 2.0f - 0.5f + gy) * stride;
                        float w = std::pow(sigmoid(det[2]) * 2.0f, 2.0f) * stride;
                        float h = std::pow(sigmoid(det[3]) * 2.0f, 2.0f) * stride;
                        result.box = {cx - w / 2.0f, cy - h / 2.0f,
                                      cx + w / 2.0f, cy + h / 2.0f};

                        // Landmarks decode
                        result.landmarks.reserve(kNumLandmarks * 2);
                        for (int k = 0; k < kNumLandmarks; ++k) {
                            int lm_base = kLandmarkOffset + k * kLandmarkStride;
                            result.landmarks.push_back(det[lm_base + 0] * stride + gx * stride);
                            result.landmarks.push_back(det[lm_base + 1] * stride + gy * stride);
                        }

                        detections.push_back(std::move(result));
                    }
                }
            }
        }
        return detections;
    }

    /**
     * @brief NMS (same O(n²) algorithm but n is now reasonable)
     */
    std::vector<YOLOv5FaceResult> apply_nms(
        const std::vector<YOLOv5FaceResult>& detections) const {
        return postprocess_utils::apply_nms(detections, nms_threshold_);
    }

    static float sigmoid(float x) { return postprocess_utils::sigmoid(x); }
};

#endif  // ANCHOR_FACE_V7_POSTPROCESSOR_HPP
