#ifndef YOLOV3TINY_PPU_POSTPROCESS_H
#define YOLOV3TINY_PPU_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief YOLOv3-Tiny PPU detection result structure
 */
struct YOLOv3TinyPPUResult {
    std::vector<float> box{};  // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};

    YOLOv3TinyPPUResult() = default;
    YOLOv3TinyPPUResult(std::vector<float> box_val, float conf, int cls_id,
                        const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }
    float iou(const YOLOv3TinyPPUResult& other) const;
    bool is_invalid(int image_width, int image_height) const;
};

/**
 * @brief YOLOv3-Tiny PPU postprocessor.
 *
 * YOLOv3-Tiny has 2 detection heads:
 *   layer_idx=0 → stride 16, anchors [[10,14],[23,27],[37,58]]
 *   layer_idx=1 → stride 32, anchors [[81,82],[135,169],[344,319]]
 *
 * Box decode (same as YOLOv5 PPU):
 *   cx = (x * 2 - 0.5 + grid_x) * stride
 *   cy = (y * 2 - 0.5 + grid_y) * stride
 *   w  = (w * w * 4) * anchor_w
 *   h  = (h * h * 4) * anchor_h
 */
class YOLOv3TinyPPUPostProcess {
public:
    YOLOv3TinyPPUPostProcess(int input_w, int input_h,
                             float obj_threshold, float score_threshold,
                             float nms_threshold);
    YOLOv3TinyPPUPostProcess();

    std::vector<YOLOv3TinyPPUResult> postprocess(const dxrt::TensorPtrs& outputs);

    void set_thresholds(float obj_threshold, float score_threshold, float nms_threshold);
    std::string get_config_info() const;

    int get_input_width()  const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_object_threshold()  const { return object_threshold_; }
    float get_score_threshold()   const { return score_threshold_; }
    float get_nms_threshold()     const { return nms_threshold_; }

private:
    int input_width_{416};
    int input_height_{416};
    float object_threshold_{0.25f};
    float score_threshold_{0.25f};
    float nms_threshold_{0.45f};
    enum { num_classes_ = 80 };

    // Anchors: layer_idx=0→stride 16, layer_idx=1→stride 32
    std::map<int, std::vector<std::pair<int, int>>> anchors_by_strides_;

    std::vector<YOLOv3TinyPPUResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv3TinyPPUResult> apply_nms(
        const std::vector<YOLOv3TinyPPUResult>& detections) const;
};

#endif  // YOLOV3TINY_PPU_POSTPROCESS_H
