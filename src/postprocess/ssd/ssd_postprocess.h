#ifndef SSD_POSTPROCESS_H
#define SSD_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief SSD detection result structure (same layout as YOLOv5Result)
 */
struct SSDResult {
    std::vector<float> box{};  // x1, y1, x2, y2 normalized in input space (0..1)
    float confidence{0.0f};
    int class_id{-1};

    SSDResult() = default;

    SSDResult(std::vector<float> box_val, float conf, int cls_id)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id) {}

    ~SSDResult() = default;

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const SSDResult& other) const;
};

/**
 * @brief SSD post-processing class
 *
 * Two-tensor output:
 *   output[0]: [1, N, num_classes+1] — class scores (softmax, class 0 = background)
 *   output[1]: [1, N, 4]             — bounding boxes (normalized ymin, xmin, ymax, xmax)
 *
 * NOTE: This postprocessor returns `SSDResult.box` as normalized coordinates
 * [x1, y1, x2, y2] in the input image space (values in range ~0.0-1.0). This
 * matches the Python postprocessors' contract and allows the caller to map
 * boxes to the original image using the preprocessing context.
 */
class SSDPostProcess {
   private:
    int input_width_{300};
    int input_height_{300};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    int num_classes_{20};
    bool has_background_{true};

    std::vector<SSDResult> apply_nms(const std::vector<SSDResult>& detections) const;

   public:
    SSDPostProcess(int input_w, int input_h,
                   float score_threshold, float nms_threshold,
                   int num_classes = 20, bool has_background = true);
    SSDPostProcess();
    ~SSDPostProcess() = default;

    std::vector<SSDResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // SSD_POSTPROCESS_H
