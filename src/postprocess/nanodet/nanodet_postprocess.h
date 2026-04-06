#ifndef NANODET_POSTPROCESS_H
#define NANODET_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief NanoDet detection result structure
 */
struct NanoDetResult {
    std::vector<float> box{};  // x1, y1, x2, y2 in input pixel space
    float confidence{0.0f};
    int class_id{-1};

    NanoDetResult() = default;

    NanoDetResult(std::vector<float> box_val, float conf, int cls_id)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id) {}

    ~NanoDetResult() = default;

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const NanoDetResult& other) const;
};

/**
 * @brief NanoDet/NanoDet-Plus post-processing with DFL (Distribution Focal Loss)
 *
 * Single-tensor output: [1, N, num_classes + 4 * (reg_max + 1)]
 * - First num_classes values: class logits (sigmoid for score)
 * - Next 4*(reg_max+1) values: DFL distribution for bbox regression
 *
 * DFL decodes (left, top, right, bottom) distances from anchor center,
 * then converts to x1y1x2y2.
 */
class NanoDetPostProcess {
   private:
    int input_width_{416};
    int input_height_{416};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    int num_classes_{80};
    int reg_max_{10};
    std::vector<int> strides_{8, 16, 32};

    // Pre-computed anchors
    std::vector<float> anchor_cx_;
    std::vector<float> anchor_cy_;
    std::vector<float> anchor_stride_;
    int total_anchors_{0};

    void build_anchors();
    std::vector<float> dfl_decode(const float* reg, int bins) const;
    std::vector<NanoDetResult> apply_nms(const std::vector<NanoDetResult>& detections) const;

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

   public:
    NanoDetPostProcess(int input_w, int input_h,
                       float score_threshold, float nms_threshold,
                       int num_classes = 80, int reg_max = 10);
    NanoDetPostProcess();
    ~NanoDetPostProcess() = default;

    std::vector<NanoDetResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // NANODET_POSTPROCESS_H
