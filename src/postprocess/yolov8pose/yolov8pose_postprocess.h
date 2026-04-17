#ifndef YOLOV8POSE_POSTPROCESS_H
#define YOLOV8POSE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

/**
 * @brief YOLOv8-Pose detection result structure
 * bbox [x1,y1,x2,y2] + confidence + 17 keypoints (x,y,conf) = 57 values
 */
struct YOLOv8PoseResult {
    std::vector<float> box{};     // x1, y1, x2, y2
    float confidence{0.0f};
    std::vector<float> landmarks{};  // 17 keypoints * 3 (x, y, conf)

    YOLOv8PoseResult() = default;

    YOLOv8PoseResult(std::vector<float> box_val, float conf, std::vector<float> kps)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(kps)) {}

    ~YOLOv8PoseResult() = default;

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const YOLOv8PoseResult& other) const;
};

/**
 * @brief YOLOv8-Pose post-processing class
 *
 * Handles anchor-free pose detection with keypoints.
 * Model output: [1, 56, N] where 56 = 4 (bbox) + 1 (score) + 51 (17*3 keypoints)
 * Transpose to [N, 56], then decode.
 */
class YOLOv8PosePostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    bool is_ort_configured_{false};

    static const int NUM_KEYPOINTS = 17;

    std::vector<YOLOv8PoseResult> decode_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv8PoseResult> apply_nms(const std::vector<YOLOv8PoseResult>& detections) const;

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

   public:
    YOLOv8PosePostProcess(int input_w, int input_h,
                          float score_threshold, float nms_threshold,
                          bool is_ort_configured = false);
    YOLOv8PosePostProcess();
    ~YOLOv8PosePostProcess() = default;

    std::vector<YOLOv8PoseResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // YOLOV8POSE_POSTPROCESS_H
