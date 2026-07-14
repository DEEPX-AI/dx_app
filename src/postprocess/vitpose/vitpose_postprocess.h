#ifndef VITPOSE_POSTPROCESS_H
#define VITPOSE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <vector>

struct VitPoseKeypoint {
    float x{0.0f};
    float y{0.0f};
    float confidence{0.0f};

    VitPoseKeypoint() = default;
    VitPoseKeypoint(float x_, float y_, float conf)
        : x(x_), y(y_), confidence(conf) {}
};

struct VitPoseResult {
    std::vector<VitPoseKeypoint> keypoints;  // 17 COCO keypoints

    VitPoseResult() = default;
    ~VitPoseResult() = default;
};

/**
 * @brief VitPose heatmap-based pose postprocessor
 *
 * Input: [1, 17, H, W] NCHW float32 heatmap (COCO 17 keypoints)
 * Output: 17 keypoints (x, y, confidence) in model output coordinates.
 *         No resize — caller handles scaling to original image size.
 */
class VitPosePostProcess {
   private:
    int input_width_{192};
    int input_height_{256};

   public:
    VitPosePostProcess(int input_w, int input_h);
    VitPosePostProcess();
    ~VitPosePostProcess() = default;

    VitPoseResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // VITPOSE_POSTPROCESS_H
