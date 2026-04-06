#ifndef CENTERPOSE_POSTPROCESS_H
#define CENTERPOSE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief CenterPose detection result
 *
 * CenterNet-based 6-DoF object pose with 8 3D bounding-box corner keypoints.
 * box:       [x1, y1, x2, y2] in model-input pixel space
 * class_id:  object class
 * landmarks: [kp0_x, kp0_y, 1.0, kp1_x, kp1_y, 1.0, ..., kp7_x, kp7_y, 1.0]
 *            8 keypoints × 3 = 24 values  (confidence=1.0 for all)
 */
struct CenterPoseResult {
    std::vector<float> box{};        // x1, y1, x2, y2
    float confidence{0.0f};
    int   class_id{0};
    std::vector<float> landmarks{};  // 8 keypoints × 3 = 24 values

    CenterPoseResult() = default;
    CenterPoseResult(std::vector<float> b, float c, int cls, std::vector<float> lm)
        : box(std::move(b)), confidence(c), class_id(cls), landmarks(std::move(lm)) {}

    float area() const {
        return (box[2] - box[0]) * (box[3] - box[1]);
    }

    float iou(const CenterPoseResult& other) const {
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
 * @brief CenterPose 6-DoF object pose estimation postprocessor
 *
 * Multi-head CenterNet outputs at output stride 4:
 *   hm        [1, C, H/4, W/4]  object center heatmap
 *   wh        [1, 2, H/4, W/4]  bbox width/height
 *   reg       [1, 2, H/4, W/4]  sub-pixel center offset
 *   hps/kps   [1, K*2, H/4, W/4]  K keypoint (x,y) offsets from center
 *   hm_hp     [1, K, H/4, W/4]   (optional) per-keypoint heatmaps
 *   hp_offset [1, 2, H/4, W/4]   (optional) keypoint sub-pixel offset
 *
 * Tensors are auto-identified by channel count.
 */
class CenterPosePostProcess {
private:
    int input_width_{512};
    int input_height_{512};
    float score_threshold_{0.3f};
    float nms_threshold_{0.5f};
    int   num_keypoints_{8};
    int   top_k_{100};
    int   stride_{4};

    // NMS helper
    std::vector<CenterPoseResult> apply_nms(const std::vector<CenterPoseResult>& det) const;

    // Pseudo-NMS via 3×3 max-pool on a single [C, H, W] heatmap.
    // Keeps only pixels that equal the local maximum (centre of the 3×3 window).
    static void heatmap_nms_inplace(std::vector<float>& hm,
                                    int C, int H, int W);

public:
    CenterPosePostProcess(int input_w, int input_h,
                          float score_threshold = 0.3f,
                          float nms_threshold   = 0.5f,
                          int   num_keypoints   = 8);
    CenterPosePostProcess();
    ~CenterPosePostProcess() = default;

    std::vector<CenterPoseResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width()  const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // CENTERPOSE_POSTPROCESS_H
