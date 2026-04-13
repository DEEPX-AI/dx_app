/**
 * @file efficientdetd4_factory.hpp
 * @brief Efficientdetd4 Abstract Factory implementation
 * 
 * Note: EfficientDet-specific SSD-style postprocessor
 */

#ifndef EFFICIENTDETD4_FACTORY_HPP
#define EFFICIENTDETD4_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/efficientdet_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Efficientdetd4Factory : public IDetectionFactory {
public:
    Efficientdetd4Factory(float score_threshold = 0.3f,
                  float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<EfficientDetPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            90  // COCO 90 classes for EfficientDet
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "Efficientdetd4"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // EFFICIENTDETD4_FACTORY_HPP
