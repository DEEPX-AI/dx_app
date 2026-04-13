#ifndef YOLACT_POSTPROCESS_H
#define YOLACT_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>
#include <array>

/**
 * @brief YOLACT instance segmentation result
 */
struct YOLACTResult {
    std::vector<float> box{};   // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};
    std::vector<float> mask{};  // binary mask values (flattened H*W)
    int mask_height{0};
    int mask_width{0};

    YOLACTResult() = default;
};

/**
 * @brief YOLACT post-processing class
 *
 * SSD-based instance segmentation with prototype masks.
 * Expects 4 output tensors:
 *   - loc:        [1, N, 4]      SSD box coordinates
 *   - conf:       [1, N, C]      class confidences
 *   - mask_coeff: [1, N, 32]     mask coefficients
 *   - proto:      [1, H, W, 32]  prototype masks
 */
class YOLACTPostProcess {
   private:
    int input_width_{550};
    int input_height_{550};
    float score_threshold_{0.3f};
    float nms_threshold_{0.5f};
    int num_classes_{80};
    bool has_background_{true};

    struct Detection {
        float x1, y1, x2, y2, score;
        int class_id, idx;
    };

    // SSD-style prior boxes (cx, cy, w, h) normalized
    std::vector<std::array<float, 4>> anchors_;

    void identifyTensors(const dxrt::TensorPtrs& outputs,
                         const dxrt::TensorPtr*& loc,
                         const dxrt::TensorPtr*& conf,
                         const dxrt::TensorPtr*& mask_coeff,
                         const dxrt::TensorPtr*& proto);

    void generateAnchors(int target_n);
    void decodeBoxes(const float* loc_data, int N,
                     std::vector<std::array<float, 4>>& decoded) const;

   public:
    YOLACTPostProcess(int input_w, int input_h,
                      float score_threshold = 0.3f,
                      float nms_threshold = 0.5f,
                      int num_classes = 80,
                      bool has_background = true);
    YOLACTPostProcess();
    ~YOLACTPostProcess() = default;

    std::vector<YOLACTResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // YOLACT_POSTPROCESS_H
