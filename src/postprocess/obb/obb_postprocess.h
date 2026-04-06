#ifndef OBB_POSTPROCESS_H
#define OBB_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief OBB (Oriented Bounding Box) result structure
 */
struct OBBResult {
    float cx{0.0f};
    float cy{0.0f};
    float width{0.0f};
    float height{0.0f};
    float angle{0.0f};     // rotation angle in radians [0, pi/2)
    float confidence{0.0f};
    int class_id{-1};

    OBBResult() = default;

    OBBResult(float cx_, float cy_, float w_, float h_,
              float angle_, float conf, int cls_id)
        : cx(cx_), cy(cy_), width(w_), height(h_),
          angle(angle_), confidence(conf), class_id(cls_id) {}

    ~OBBResult() = default;
};

/**
 * @brief OBB post-processing class for oriented bounding box detection
 *
 * Model output: [N, 7] or [1, N, 7]
 * Format: [cx, cy, w, h, score, class_id, angle]
 *
 * Applies angle regularization: if angle >= pi/2, swap w/h and normalize.
 */
class OBBPostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    float score_threshold_{0.3f};

    void regularize_angle(float& w, float& h, float& angle) const;

   public:
    OBBPostProcess(int input_w, int input_h, float score_threshold = 0.3f);
    OBBPostProcess();
    ~OBBPostProcess() = default;

    std::vector<OBBResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // OBB_POSTPROCESS_H
