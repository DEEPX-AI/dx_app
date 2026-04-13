#ifndef YOLOX_PPU_POSTPROCESS_H
#define YOLOX_PPU_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOX PPU detection result structure
 */
struct YOLOXPPUResult {
    std::vector<float> box{};  // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};

    YOLOXPPUResult() = default;
    YOLOXPPUResult(std::vector<float> box_val, float conf, int cls_id,
                   const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }
    float iou(const YOLOXPPUResult& other) const;
    bool is_invalid(int image_width, int image_height) const;
};

/**
 * @brief YOLOX PPU postprocessor — anchor-free grid-offset decode.
 *
 * DeviceBoundingBox_t fields used:
 *   x, y   — raw tx, ty offset from grid cell (NOT post-sigmoid)
 *   w, h   — raw tw, th (exponentiated to get width/height)
 *   grid_x, grid_y — grid cell index
 *   layer_idx — stride index: 0→8, 1→16, 2→32
 *   score   — detection score
 *   label   — class id
 *
 * Decode:
 *   cx = (tx + grid_x) * stride
 *   cy = (ty + grid_y) * stride
 *   w  = exp(tw) * stride
 *   h  = exp(th) * stride
 */
class YOLOXPPUPostProcess {
public:
    YOLOXPPUPostProcess(int input_w, int input_h,
                        float obj_threshold, float score_threshold,
                        float nms_threshold);
    YOLOXPPUPostProcess();

    std::vector<YOLOXPPUResult> postprocess(const dxrt::TensorPtrs& outputs);

    void set_thresholds(float obj_threshold, float score_threshold, float nms_threshold);
    std::string get_config_info() const;

    int get_input_width()  const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_object_threshold()  const { return object_threshold_; }
    float get_score_threshold()   const { return score_threshold_; }
    float get_nms_threshold()     const { return nms_threshold_; }

private:
    int input_width_{640};
    int input_height_{640};
    float object_threshold_{0.25f};
    float score_threshold_{0.25f};
    float nms_threshold_{0.45f};

    // Strides indexed by layer_idx: 0→8, 1→16, 2→32
    static constexpr int NUM_STRIDES = 3;
    static constexpr int STRIDES[NUM_STRIDES] = {8, 16, 32};

    std::vector<YOLOXPPUResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOXPPUResult> apply_nms(const std::vector<YOLOXPPUResult>& detections) const;
};

#endif  // YOLOX_PPU_POSTPROCESS_H
