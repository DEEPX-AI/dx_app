#ifndef PPMATTING_POSTPROCESS_H
#define PPMATTING_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <vector>

/**
 * @brief An alpha matte at the model's output resolution, row-major, in [0, 1]
 *
 * The caller resizes it to the image and thresholds it into the foreground
 * class map, as the Python PPMattingPostprocessor does.
 */
struct PPMattingResult {
    std::vector<float> alpha{};
    int width{0};
    int height{0};

    PPMattingResult() = default;
    ~PPMattingResult() = default;
};

/**
 * @brief PP-Matting post-processing
 *
 * Takes the single-channel alpha map, (1, 1, H, W), (1, H, W, 1), (1, H, W) or
 * (H, W); applies a sigmoid only when the map holds logits; clips to [0, 1].
 * A multi-class segmentation head is refused on purpose.
 */
class PPMattingPostProcess {
   private:
    int input_width_{512};
    int input_height_{512};

   public:
    PPMattingPostProcess(int input_w = 512, int input_h = 512);
    ~PPMattingPostProcess() = default;

    PPMattingResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // PPMATTING_POSTPROCESS_H
