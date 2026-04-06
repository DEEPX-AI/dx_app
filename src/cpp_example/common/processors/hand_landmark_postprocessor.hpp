/**
 * @file hand_landmark_postprocessor.hpp
 * @brief MediaPipe-style Hand Landmark postprocessor
 *
 * Processes hand landmark model output (e.g., MediaPipe HandLandmarkLite).
 *
 * Expected output tensors:
 *   - Tensor[0]: [1, 63] — 21 landmarks × 3 (x, y, z) normalized coordinates
 *   - Tensor[1]: [1, 1]  — handedness score (optional)
 *   - Tensor[2]: [1, 1]  — hand confidence score (optional)
 *
 * 21 keypoints follow the MediaPipe hand landmark topology:
 *   0: Wrist
 *   1-4: Thumb (CMC, MCP, IP, TIP)  
 *   5-8: Index finger (MCP, PIP, DIP, TIP)
 *   9-12: Middle finger
 *   13-16: Ring finger
 *   17-20: Pinky finger
 */

#ifndef DXAPP_HAND_LANDMARK_POSTPROCESSOR_HPP
#define DXAPP_HAND_LANDMARK_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace dxapp {

class HandLandmarkPostprocessor : public IPostprocessor<HandLandmarkResult> {
public:
    static constexpr int NUM_LANDMARKS = 21;
    static constexpr int COORDS_PER_LANDMARK = 3;  // x, y, z

    HandLandmarkPostprocessor(int input_width, int input_height, float confidence_threshold = 0.5f)
        : input_width_(input_width), input_height_(input_height), confidence_threshold_(confidence_threshold) {}

    std::vector<HandLandmarkResult> process(
        const dxrt::TensorPtrs& outputs,
        const PreprocessContext& ctx) override {

        std::vector<HandLandmarkResult> results;
        if (outputs.empty()) return results;

        HandLandmarkResult result;

        // Parse 21 landmarks from primary output tensor
        auto lm_tensor = outputs[0];
        const float* lm_data = static_cast<const float*>(lm_tensor->data());
        int num_elements = NUM_LANDMARKS * COORDS_PER_LANDMARK;

        int num_lmks = std::min(NUM_LANDMARKS, num_elements / COORDS_PER_LANDMARK);
        result.landmarks.resize(num_lmks);

        float scale_x = static_cast<float>(ctx.original_width);
        float scale_y = static_cast<float>(ctx.original_height);

        for (int i = 0; i < num_lmks; ++i) {
            int offset = i * COORDS_PER_LANDMARK;
            float x = lm_data[offset + 0];
            float y = lm_data[offset + 1];
            // z is depth (relative), we store in confidence for now
            float z = (offset + 2 < num_elements) ? lm_data[offset + 2] : 0.0f;

            // Coordinates may be normalized [0, 1] or in input pixel space
            if (x <= 1.0f && y <= 1.0f && x >= 0.0f && y >= 0.0f) {
                // Normalized → scale to original image
                result.landmarks[i] = Keypoint(x * scale_x, y * scale_y, z);
            } else {
                // Input pixel space → scale to original image
                float sx = scale_x / input_width_;
                float sy = scale_y / input_height_;
                result.landmarks[i] = Keypoint(x * sx, y * sy, z);
            }
        }

        // Parse hand presence score (tensor[1]) — used as confidence
        if (outputs.size() > 1) {
            const float* presence_data = static_cast<const float*>(outputs[1]->data());
            result.confidence = presence_data[0];
        } else {
            result.handedness = "Unknown";
            result.confidence = 1.0f;
        }

        // Parse handedness if available (tensor[2])
        if (outputs.size() > 2) {
            const float* hand_data = static_cast<const float*>(outputs[2]->data());
            float handedness_score = hand_data[0];
            result.handedness = (handedness_score > 0.5f) ? "Right" : "Left";
        }

        // Filter by confidence threshold
        if (result.confidence >= confidence_threshold_) {
            results.push_back(result);
        }
        return results;
    }

    std::string getModelName() const override { return "HandLandmark"; }

private:
    int input_width_;
    int input_height_;
    float confidence_threshold_;
};

}  // namespace dxapp

#endif  // DXAPP_HAND_LANDMARK_POSTPROCESSOR_HPP
