#ifndef SEMANTIC_SEG_POSTPROCESS_H
#define SEMANTIC_SEG_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief Generic semantic segmentation result structure
 * Compatible with DeepLabv3, BiSeNetV1, BiSeNetV2, and other semantic segmentation models
 */
struct SemanticSegResult {
    std::vector<int> segmentation_mask;  // Class IDs per pixel (H*W)
    int width{0};
    int height{0};
    int num_classes{0};

    SemanticSegResult() = default;

    SemanticSegResult(const std::vector<int>& seg_mask, int w, int h)
        : segmentation_mask(seg_mask), width(w), height(h) {}

    ~SemanticSegResult() = default;
};

/**
 * @brief Generic semantic segmentation post-processing class
 *
 * Handles argmax-based segmentation for any model that outputs class logits.
 * Supports both NCHW [1, C, H, W] and NHWC [1, H, W, C] output layouts.
 * Works with any number of classes (not hardcoded like DeepLabv3PostProcess).
 */
class SemanticSegPostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    int num_classes_{0};  // 0 = auto-detect from tensor shape

    std::vector<int> apply_argmax_nchw(const float* data, int C, int H, int W) const;
    std::vector<int> apply_argmax_nhwc(const float* data, int H, int W, int C) const;

   public:
    /**
     * @brief Constructor
     * @param input_w Model input width
     * @param input_h Model input height
     * @param num_classes Number of classes (0 = auto-detect from output shape)
     */
    SemanticSegPostProcess(int input_w, int input_h, int num_classes = 0);
    SemanticSegPostProcess();
    ~SemanticSegPostProcess() = default;

    SemanticSegResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    int get_num_classes() const { return num_classes_; }
};

#endif  // SEMANTIC_SEG_POSTPROCESS_H
