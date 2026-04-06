#ifndef DAMOYOLO_POSTPROCESS_H
#define DAMOYOLO_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief DamoYOLO detection result structure
 */
struct DamoYOLOResult {
    std::vector<float> box{};  // x1, y1, x2, y2 in input pixel space
    float confidence{0.0f};
    int class_id{-1};

    DamoYOLOResult() = default;

    DamoYOLOResult(std::vector<float> box_val, float conf, int cls_id)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id) {}

    ~DamoYOLOResult() = default;

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const DamoYOLOResult& other) const;
};

/**
 * @brief DamoYOLO post-processing class
 *
 * Two-tensor split-head output:
 *   output[0]: [1, N, num_classes] — class scores (already sigmoid'd)
 *   output[1]: [1, N, 4]          — bounding boxes (x1, y1, x2, y2 in pixel scale)
 */
class DamoYOLOPostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    int num_classes_{80};

    std::vector<DamoYOLOResult> apply_nms(const std::vector<DamoYOLOResult>& detections) const;

   public:
    DamoYOLOPostProcess(int input_w, int input_h,
                        float score_threshold, float nms_threshold,
                        int num_classes = 80);
    DamoYOLOPostProcess();
    ~DamoYOLOPostProcess() = default;

    std::vector<DamoYOLOResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // DAMOYOLO_POSTPROCESS_H
