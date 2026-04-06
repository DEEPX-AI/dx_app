#ifndef HAND_LANDMARK_POSTPROCESS_H
#define HAND_LANDMARK_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief Single hand keypoint
 */
struct HandKeypoint {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    HandKeypoint() = default;
    HandKeypoint(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

/**
 * @brief Hand landmark detection result
 */
struct HandLandmarkResult {
    std::vector<HandKeypoint> landmarks;  // 21 hand landmarks
    float confidence{0.0f};              // hand presence confidence
    std::string handedness;              // "Left", "Right", or "Unknown"

    HandLandmarkResult() = default;
    ~HandLandmarkResult() = default;
};

/**
 * @brief Hand landmark post-processing class
 *
 * Processes hand landmark model output (e.g., MediaPipe HandLandmarkLite).
 *
 * Expected output tensors:
 *   - Tensor[0]: [1, 63] — 21 landmarks x 3 (x, y, z) normalized coordinates
 *   - Tensor[1]: [1, 1]  — handedness score (optional)
 *   - Tensor[2]: [1, 1]  — hand confidence score (optional)
 */
class HandLandmarkPostProcess {
   private:
    int input_width_{224};
    int input_height_{224};
    float confidence_threshold_{0.5f};

    static constexpr int NUM_LANDMARKS = 21;
    static constexpr int COORDS_PER_LANDMARK = 3;

   public:
    HandLandmarkPostProcess(int input_w, int input_h, float confidence_threshold = 0.5f);
    HandLandmarkPostProcess();
    ~HandLandmarkPostProcess() = default;

    HandLandmarkResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_confidence_threshold() const { return confidence_threshold_; }
    void set_confidence_threshold(float threshold) { confidence_threshold_ = threshold; }
};

#endif  // HAND_LANDMARK_POSTPROCESS_H
