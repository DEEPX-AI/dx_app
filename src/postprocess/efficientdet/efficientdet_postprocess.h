#ifndef EFFICIENTDET_POSTPROCESS_H
#define EFFICIENTDET_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief EfficientDet detection result
 */
struct EfficientDetResult {
    std::vector<float> box{};   // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};

    EfficientDetResult() = default;
    EfficientDetResult(std::vector<float> b, float c, int cls)
        : box(std::move(b)), confidence(c), class_id(cls) {}
};

/**
 * @brief EfficientDet post-processing class
 *
 * Handles:
 *   - TF format (4 tensors): [boxes, classes, scores, num_detections]
 *   - 2-tensor format: [boxes(N,4), scores(N,C)] or [scores, boxes]
 */
class EfficientDetPostProcess {
   private:
    int input_width_{512};
    int input_height_{512};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    int num_classes_{90};
    bool has_background_{true};

    std::vector<EfficientDetResult> processTFFormat(const dxrt::TensorPtrs& outputs);
    std::vector<EfficientDetResult> process2Tensor(const dxrt::TensorPtrs& outputs);

   public:
    EfficientDetPostProcess(int input_w, int input_h,
                            float score_threshold = 0.3f,
                            float nms_threshold = 0.45f,
                            int num_classes = 90,
                            bool has_background = true);
    EfficientDetPostProcess();
    ~EfficientDetPostProcess() = default;

    std::vector<EfficientDetResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // EFFICIENTDET_POSTPROCESS_H
