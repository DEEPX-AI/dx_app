#ifndef ULFGFD_POSTPROCESS_H
#define ULFGFD_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief ULFGFD detection result
 *
 * Ultra-Light-Fast-Generic Face Detector: SSD-style, no landmarks.
 * box: [x1, y1, x2, y2] in model-input pixel space.
 */
struct ULFGFDResult {
    std::vector<float> box{};
    float confidence{0.0f};

    ULFGFDResult() = default;
    ULFGFDResult(std::vector<float> b, float c) : box(std::move(b)), confidence(c) {}

    float area() const {
        return (box[2] - box[0]) * (box[3] - box[1]);
    }

    float iou(const ULFGFDResult& other) const {
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
 * @brief ULFGFD (Ultra-Light-Fast-Generic Face Detector) postprocessor
 *
 * Handles SSD-style face detection without landmarks.
 * Model outputs 2 tensors:
 *   - scores: [1, N, 2]  (background/face scores)
 *   - boxes:  [1, N, 4]  (normalized [x1,y1,x2,y2] in [0,1])
 */
class ULFGFDPostProcess {
private:
    int input_width_{320};
    int input_height_{240};
    float score_threshold_{0.7f};
    float nms_threshold_{0.3f};
    int top_k_{200};

    std::vector<ULFGFDResult> apply_nms(const std::vector<ULFGFDResult>& detections) const;

public:
    ULFGFDPostProcess(int input_w, int input_h,
                      float score_threshold = 0.7f,
                      float nms_threshold = 0.3f);
    ULFGFDPostProcess();
    ~ULFGFDPostProcess() = default;

    std::vector<ULFGFDResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
};

#endif  // ULFGFD_POSTPROCESS_H
