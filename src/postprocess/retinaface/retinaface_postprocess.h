#ifndef RETINAFACE_POSTPROCESS_H
#define RETINAFACE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <array>
#include <string>
#include <vector>

/**
 * @brief RetinaFace detection result
 *
 * Anchor-based face detection with 5-point facial landmarks.
 * box:       [x1, y1, x2, y2] in model-input pixel space
 * landmarks: [kp0_x, kp0_y, kp1_x, kp1_y, ..., kp4_x, kp4_y]  (10 values)
 */
struct RetinaFaceResult {
    std::vector<float> box{};        // x1, y1, x2, y2
    float confidence{0.0f};
    std::vector<float> landmarks{};  // 5 keypoints × 2 = 10 values

    RetinaFaceResult() = default;
    RetinaFaceResult(std::vector<float> b, float c, std::vector<float> lm)
        : box(std::move(b)), confidence(c), landmarks(std::move(lm)) {}

    float area() const {
        return (box[2] - box[0]) * (box[3] - box[1]);
    }

    float iou(const RetinaFaceResult& other) const {
        float ix1 = std::max(box[0], other.box[0]);
        float iy1 = std::max(box[1], other.box[1]);
        float ix2 = std::min(box[2], other.box[2]);
        float iy2 = std::min(box[3], other.box[3]);
        float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
        float uni = area() + other.area() - inter;
        return (uni > 0.0f) ? inter / uni : 0.0f;
    }
};

/**
 * @brief RetinaFace face detection postprocessor
 *
 * Generates multi-scale anchors, decodes bbox deltas and 5-point landmarks,
 * applies softmax scoring and NMS.
 *
 * Expected model outputs (sorted by last dim, auto-detected):
 *   bbox_delta:  [1, N, 4]   (dx, dy, dw, dh anchor deltas)
 *   scores:      [1, N, 2]   (background / face logits)
 *   landmarks:   [1, N, 10]  (5 keypoints × 2 offset deltas)
 *
 * Anchor params (MobileNet0.25 640×640 defaults):
 *   strides   = {8, 16, 32}
 *   min_sizes = {{16,32}, {64,128}, {256,512}}
 *   variance  = {0.1, 0.2}
 */
class RetinaFacePostProcess {
private:
    int input_width_{640};
    int input_height_{640};
    float score_threshold_{0.5f};
    float nms_threshold_{0.4f};
    int   top_k_{750};

    // Anchor configuration
    std::vector<int> strides_{8, 16, 32};
    std::vector<std::vector<int>> min_sizes_{{16, 32}, {64, 128}, {256, 512}};
    float var0_{0.1f};  // centre-offset variance
    float var1_{0.2f};  // size variance

    // Cached anchor array: each row = [cx_norm, cy_norm, sw_norm, sh_norm]
    mutable std::vector<std::array<float, 4>> priors_;

    void generate_priors() const;
    std::vector<RetinaFaceResult> apply_nms(const std::vector<RetinaFaceResult>& det) const;

    // Identify bbox / score / landmark tensors by last dimension (4, 2, 10).
    struct IdentifiedTensors_ {
        dxrt::Tensor* bbox = nullptr;
        dxrt::Tensor* score = nullptr;
        dxrt::Tensor* landmark = nullptr;
    };
    IdentifiedTensors_ identifyTensors_(const dxrt::TensorPtrs& outputs) const;

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    static float softmax2_face(float bg, float fg);  // softmax on 2-class logit pair

public:
    RetinaFacePostProcess(int input_w, int input_h,
                          float score_threshold = 0.5f,
                          float nms_threshold   = 0.4f);
    RetinaFacePostProcess();
    ~RetinaFacePostProcess() = default;

    std::vector<RetinaFaceResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width()  const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // RETINAFACE_POSTPROCESS_H
