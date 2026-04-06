#ifndef YOLOV5_SEG_POSTPROCESS_H
#define YOLOV5_SEG_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOv5-seg detection result structure
 * Contains bounding box, confidence, class info, and instance segmentation mask
 *
 * YOLOv5-seg differs from YOLOv8-seg in that it has an objectness score:
 *   output0: [1, N, 117] = [cx, cy, w, h, obj, 80 classes, 32 mask_coefs]
 *   output1: [1, 32, mask_h, mask_w] = mask prototypes
 */
struct YOLOv5SegResult {
    std::vector<float> box{};          // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};
    std::vector<float> seg_mask_coef{};  // 32 mask coefficients
    std::vector<float> mask{};           // Binary segmentation mask (flattened H*W)
    int mask_height{0};
    int mask_width{0};

    YOLOv5SegResult() = default;

    YOLOv5SegResult(std::vector<float> box_val, float conf, int cls_id,
                     const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    ~YOLOv5SegResult() = default;
    YOLOv5SegResult(const YOLOv5SegResult&) = default;
    YOLOv5SegResult& operator=(const YOLOv5SegResult&) = default;
    YOLOv5SegResult(YOLOv5SegResult&&) noexcept = default;
    YOLOv5SegResult& operator=(YOLOv5SegResult&&) noexcept = default;

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }
    bool is_invalid(int image_width, int image_height) const;
};

/**
 * @brief YOLOv5-seg post-processing class
 *
 * Handles YOLOv5-seg output format with objectness score:
 *   output0: [1, N, 117] -> [cx, cy, w, h, obj, 80 classes, 32 mask_coefs]
 *   output1: [1, 32, mask_h, mask_w] -> mask prototypes
 *
 * Key difference from YOLOv8-seg:
 *   - Objectness score present (index 4)
 *   - Output is NOT transposed (row-major per anchor)
 *   - Final confidence = objectness * class_score
 */
class YOLOv5SegPostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    float obj_threshold_{0.25f};
    float score_threshold_{0.5f};
    float nms_threshold_{0.45f};
    enum { num_classes_ = 80 };
    enum { num_mask_coefs_ = 32 };

    bool is_ort_configured_{false};

    std::vector<YOLOv5SegResult> decoding_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv5SegResult> apply_nms(
        const std::vector<YOLOv5SegResult>& detections) const;
    void decode_masks(const dxrt::TensorPtrs& outputs,
                      std::vector<YOLOv5SegResult>& detections);

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

   public:
    YOLOv5SegPostProcess(int input_w, int input_h, float obj_threshold,
                          float score_threshold, float nms_threshold,
                          bool is_ort_configured = true);
    YOLOv5SegPostProcess();
    ~YOLOv5SegPostProcess() = default;

    std::vector<YOLOv5SegResult> postprocess(const dxrt::TensorPtrs& outputs);

    void set_thresholds(float obj_threshold, float score_threshold, float nms_threshold);
    std::string get_config_info() const;

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_obj_threshold() const { return obj_threshold_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    bool get_is_ort_configured() const { return is_ort_configured_; }
    static int get_num_classes() { return num_classes_; }
};

#endif  // YOLOV5_SEG_POSTPROCESS_H
