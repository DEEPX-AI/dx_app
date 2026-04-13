/**
 * @file ssdvgg16_factory.hpp
 * @brief SSD MobileNet V1 Abstract Factory implementation
 * 
 * Uses v3-native SSD postprocessor (no legacy postprocess lib).
 */

#ifndef SSDVGG16_FACTORY_HPP
#define SSDVGG16_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/ssd_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class SSDvgg16Factory : public IDetectionFactory {
public:
    SSDvgg16Factory(float score_threshold = 0.3f,
                  float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<SSDPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "SSDvgg16"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // SSDVGG16_FACTORY_HPP
