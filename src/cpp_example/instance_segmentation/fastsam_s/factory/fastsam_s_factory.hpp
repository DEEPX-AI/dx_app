/**
 * @file fastsam_s_factory.hpp
 * @brief Fastsam_s Abstract Factory implementation
 */

#ifndef FASTSAM_S_FACTORY_HPP
#define FASTSAM_S_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Fastsam_sFactory : public IInstanceSegmentationFactory {
public:
    Fastsam_sFactory(float score_threshold = 0.5f,
                      float nms_threshold = 0.65f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<InstanceSegmentationResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<YOLOv8SegPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            is_ort_configured,
            1  // FastSAM: class-agnostic (1 class)
        );
    }

    VisualizerPtr<InstanceSegmentationResult> createVisualizer() override {
        return std::make_unique<InstanceSegmentationVisualizer>(false);  // No boxes for FastSAM
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "Fastsam_s"; }
    std::string getTaskType() const override { return "instance_segmentation"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // FASTSAM_S_FACTORY_HPP
