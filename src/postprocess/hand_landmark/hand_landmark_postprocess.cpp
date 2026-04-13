#include "hand_landmark_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// C++14 requires out-of-class definitions for static constexpr members that are ODR-used
constexpr int HandLandmarkPostProcess::NUM_LANDMARKS;
constexpr int HandLandmarkPostProcess::COORDS_PER_LANDMARK;

HandLandmarkPostProcess::HandLandmarkPostProcess(int input_w, int input_h, float confidence_threshold)
    : input_width_(input_w), input_height_(input_h), confidence_threshold_(confidence_threshold) {}

HandLandmarkPostProcess::HandLandmarkPostProcess()
    : input_width_(224), input_height_(224), confidence_threshold_(0.5f) {}

HandLandmarkResult HandLandmarkPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    HandLandmarkResult result;
    result.handedness = "Unknown";
    result.confidence = 0.0f;

    if (outputs.empty()) {
        return result;
    }

    // Parse 21 landmarks from primary output tensor [1, 63]
    const float* lm_data = static_cast<const float*>(outputs[0]->data());
    const auto& shape = outputs[0]->shape();

    int num_elements = 1;
    for (const auto& s : shape) {
        num_elements *= static_cast<int>(s);
    }

    int num_lmks = std::min(NUM_LANDMARKS, num_elements / COORDS_PER_LANDMARK);
    result.landmarks.resize(num_lmks);

    for (int i = 0; i < num_lmks; ++i) {
        int offset = i * COORDS_PER_LANDMARK;
        float x = lm_data[offset + 0];
        float y = lm_data[offset + 1];
        float z = (offset + 2 < num_elements) ? lm_data[offset + 2] : 0.0f;

        // Coordinates normalized [0, 1] relative to input dimensions
        if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f) {
            result.landmarks[i] = HandKeypoint(x, y, z);
        } else {
            // Input pixel space — normalize to [0, 1]
            result.landmarks[i] = HandKeypoint(
                x / static_cast<float>(input_width_),
                y / static_cast<float>(input_height_),
                z);
        }
    }

    // Parse hand presence score (tensor[1]) — used as confidence
    if (outputs.size() > 1) {
        const float* presence_data = static_cast<const float*>(outputs[1]->data());
        result.confidence = presence_data[0];
    }

    // Parse handedness if available (tensor[2])
    if (outputs.size() > 2) {
        const float* hand_data = static_cast<const float*>(outputs[2]->data());
        float handedness_score = hand_data[0];
        result.handedness = (handedness_score > 0.5f) ? "Right" : "Left";
    }

    // Filter by confidence threshold
    if (result.confidence < confidence_threshold_) {
        result.landmarks.clear();
        result.handedness = "Unknown";
    }

    return result;
}
