#ifndef DEPTH_POSTPROCESS_H
#define DEPTH_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief Depth estimation result structure
 */
struct DepthResult {
    std::vector<uint8_t> depth_map;  // Normalized depth map (0-255) in HW order
    int width{0};
    int height{0};

    DepthResult() = default;

    DepthResult(const std::vector<uint8_t>& map, int w, int h)
        : depth_map(map), width(w), height(h) {}

    ~DepthResult() = default;
};

/**
 * @brief Depth estimation post-processing class
 *
 * Handles depth estimation models that output [1, 1, H, W] or [1, H, W].
 * Squeezes to [H, W], normalizes to 0-255 uint8.
 */
class DepthPostProcess {
   private:
    int input_width_{224};
    int input_height_{224};

   public:
    /**
     * @brief Constructor
     * @param input_w Model input width
     * @param input_h Model input height
     */
    DepthPostProcess(int input_w, int input_h);
    DepthPostProcess();
    ~DepthPostProcess() = default;

    DepthResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // DEPTH_POSTPROCESS_H
