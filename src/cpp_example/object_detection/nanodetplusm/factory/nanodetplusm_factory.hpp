/**
 * @file nanodetplusm_factory.hpp
 * @brief NanoDetplusm Abstract Factory implementation
 * 
 * Uses v3-native NanoDet postprocessor with DFL decoding.
 */

#ifndef NANODETPLUSM_FACTORY_HPP
#define NANODETPLUSM_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/nanodet_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class NanoDetplusmFactory : public IDetectionFactory {
public:
    NanoDetplusmFactory(float score_threshold = 0.35f,
                   float nms_threshold = 0.6f,
                   int reg_max = 7)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold),
          reg_max_(reg_max) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<NanoDetPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_, 80, reg_max_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        reg_max_ = config.get<int>("reg_max", reg_max_);
    }

    std::string getModelName() const override { return "NanoDetplusm"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
    int reg_max_;
};

}  // namespace dxapp

#endif  // NANODETPLUSM_FACTORY_HPP
