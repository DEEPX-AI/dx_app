/**
 * @file mediapipe_hand_detector_factory.hpp
 * @brief MediapipeHandDetectorFactory Abstract Factory implementation
 *
 * MediaPipe Palm/Hand Detector (192x192, UINT8 RGB):
 *   - Input:  uint8 RGB, direct resize (no letterbox)
 *   - Output: [1,2016,18] regression + [1,2016,1] score logits
 */

#ifndef MEDIAPIPE_HAND_DETECTOR_FACTORY_HPP
#define MEDIAPIPE_HAND_DETECTOR_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/mediapipe_hand_postprocessor.hpp"
#include "common/visualizers/face_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class MediapipeHandDetectorFactory : public IFaceDetectionFactory {
public:
    MediapipeHandDetectorFactory(float score_threshold = 0.5f, float nms_threshold = 0.3f)
        : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // UINT8 RGB — default SimpleResizePreprocessor (BGR→RGB, no float conversion)
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<FaceDetectionResult> createPostprocessor(int input_width, int /*input_height*/, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        // input_width == input_height == 192 (square model)
        return std::make_unique<MediaPipeHandPostprocessor>(
            input_width, score_threshold_, nms_threshold_);
    }

    VisualizerPtr<FaceDetectionResult> createVisualizer() override {
        return std::make_unique<FaceVisualizer>("Hand");
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_   = config.get<float>("nms_threshold",   nms_threshold_);
    }

    std::string getModelName() const override { return "MediaPipe Hand Detector"; }
    std::string getTaskType() const override { return "hand_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // MEDIAPIPE_HAND_DETECTOR_FACTORY_HPP
