#ifndef YOLOV4_POSTPROCESS_H
#define YOLOV4_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <vector>

/**
 * @brief YOLOv4 detection result structure
 */
struct YOLOv4Result {
    std::vector<float> box{};  // x1, y1, x2, y2 in input pixel space
    float confidence{0.0f};
    int class_id{-1};

    YOLOv4Result() = default;

    YOLOv4Result(std::vector<float> box_val, float conf, int cls_id)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id) {}

    ~YOLOv4Result() = default;

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const YOLOv4Result& other) const;
};

/**
 * @brief YOLOv4 (DarkNet) post-processing class
 *
 * Two-tensor split-head output (model already decodes boxes):
 *   - boxes  [1, N, (1,) 4] — bounding boxes (x1, y1, x2, y2)
 *   - scores [1, N, num_classes] — per-class scores (no objectness column)
 *
 * The boxes/scores tensor order is auto-detected via the last dimension
 * (the tensor whose last dim == 4 is the boxes tensor).
 *
 * When `normalized` is true (default), box coordinates are assumed to be in
 * the normalized [0, 1] range and are scaled by the input width/height to
 * input-pixel space. This matches the golden YOLOv5Postprocessor's
 * `_process_separate_boxes_confs` path (xyxy box format + normalized scaling)
 * and uses global (cross-class) NMS via cv2.dnn.NMSBoxes semantics.
 */
class YOLOv4PostProcess {
   private:
    int input_width_{512};
    int input_height_{512};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    int num_classes_{80};
    bool normalized_{true};

    std::vector<YOLOv4Result> apply_nms(
        const std::vector<YOLOv4Result>& detections) const;

   public:
    YOLOv4PostProcess(int input_w, int input_h,
                      float score_threshold, float nms_threshold,
                      int num_classes = 80, bool normalized = true);
    YOLOv4PostProcess();
    ~YOLOv4PostProcess() = default;

    std::vector<YOLOv4Result> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // YOLOV4_POSTPROCESS_H
