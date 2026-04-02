#ifndef DNCNN_POSTPROCESS_H
#define DNCNN_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief DnCNN image restoration result structure
 * Contains the denoised/restored image data
 */
struct DnCNNResult {
    std::vector<float> image{};  // Flattened restored image (H*W), values in [0, 1]
    int height{0};               // Image height
    int width{0};                // Image width

    DnCNNResult() = default;

    DnCNNResult(std::vector<float> img, int h, int w)
        : image(std::move(img)), height(h), width(w) {}

    ~DnCNNResult() = default;
    DnCNNResult(const DnCNNResult&) = default;
    DnCNNResult& operator=(const DnCNNResult&) = default;
    DnCNNResult(DnCNNResult&&) noexcept = default;
    DnCNNResult& operator=(DnCNNResult&&) noexcept = default;

};

/**
 * @brief DnCNN post-processing class
 *
 * DnCNN outputs a noise residual in [1, C, H, W] format.
 * For single-channel denoising: [1, 1, H, W]
 *
 * Since the C++ postprocessor does not have access to the original
 * normalized input, this class simply clips and normalizes the residual
 * output to produce a usable image.
 *
 * For full denoised = input - residual computation, the caller should
 * subtract the residual from the normalized input externally.
 */
class DnCNNPostProcess {
   private:
    int input_width_{50};
    int input_height_{50};

   public:
    /**
     * @brief Constructor
     * @param input_w Model input width
     * @param input_h Model input height
     */
    DnCNNPostProcess(int input_w, int input_h);
    DnCNNPostProcess();
    ~DnCNNPostProcess() = default;

    /**
     * @brief Process DnCNN model outputs
     * @param outputs Vector of output tensors from the model
     * @return DnCNNResult with the output image data
     *
     * The output tensor is expected to be [1, C, H, W] float.
     * Returns the residual/output clipped to [0, 1] and flattened to H*W.
     */
    DnCNNResult postprocess(const dxrt::TensorPtrs& outputs);

    // Getters
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // DNCNN_POSTPROCESS_H
