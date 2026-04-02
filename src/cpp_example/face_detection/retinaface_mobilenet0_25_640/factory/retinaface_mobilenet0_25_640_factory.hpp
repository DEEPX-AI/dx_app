/**
 * @file retinaface_mobilenet0_25_640_factory.hpp
 * @brief RetinaFace MobileNet0.25 Abstract Factory for face detection
 *
 * Part of DX-APP v3.0.0 refactoring.
 * Anchor-based face detection with 5-point landmarks.
 */

#ifndef RETINAFACE_MOBILENET0_25_640_FACTORY_HPP
#define RETINAFACE_MOBILENET0_25_640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/retinaface_postprocessor.hpp"
#include "common/visualizers/face_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class RetinaFaceFactory : public IFaceDetectionFactory {
public:
    RetinaFaceFactory(float score_threshold = 0.5f,
                      float nms_threshold = 0.4f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<FaceDetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
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

    std::string getModelName() const override { return "RetinaFace-MobileNet0.25"; }
    std::string getTaskType() const override { return "face_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // RETINAFACE_MOBILENET0_25_640_FACTORY_HPP
