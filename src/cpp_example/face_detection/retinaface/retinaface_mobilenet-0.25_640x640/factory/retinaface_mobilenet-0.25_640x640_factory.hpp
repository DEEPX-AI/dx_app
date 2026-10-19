/**
 * @file retinaface_mobilenet-0.25_640x640_factory.hpp
 * @brief RetinaFace MobileNet0.25 Abstract Factory for face detection
 *
 * Anchor-based face detection with 5-point landmarks.
 */

#ifndef RETINAFACE_MOBILENET_0_25_640X640_FACTORY_HPP
#define RETINAFACE_MOBILENET_0_25_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/retinaface_postprocessor.hpp"
#include "common/visualizers/face_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/simple_resize_preprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_retinaface_mobilenet_0_25_640x640 {

class RetinafaceFactory : public IFaceDetectionFactory {
public:

    RetinafaceFactory(float score_threshold = 0.5f,
                      float nms_threshold = 0.4f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // default: retinaface_mobilenet-0.25_640x640
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<FaceDetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // default: retinaface_mobilenet-0.25_640x640
        (void)is_ort_configured;
        return std::make_unique<RetinaFacePostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_
        );
    }

    VisualizerPtr<FaceDetectionResult> createVisualizer() override {
        return std::make_unique<FaceVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "retinaface_mobilenet-0.25_640x640";
    }
    std::string getTaskType() const override { return "face_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace v_retinaface_mobilenet_0_25_640x640
}  // namespace dxapp

#endif  // RETINAFACE_MOBILENET_0_25_640X640_FACTORY_HPP
