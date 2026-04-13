/**
 * @file yolo26s_seg_factory.hpp
 * @brief Yolo26s_seg Abstract Factory implementation
 */

#ifndef YOLO26S_SEG_FACTORY_HPP
#define YOLO26S_SEG_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Yolo26s_segFactory : public IInstanceSegmentationFactory {
public:
    Yolo26s_segFactory(float score_threshold = 0.3f,
                      float nms_threshold = 0.45f)
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
            is_ort_configured
        );
    }

    VisualizerPtr<InstanceSegmentationResult> createVisualizer() override {
        return std::make_unique<InstanceSegmentationVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "Yolo26s_seg"; }
    std::string getTaskType() const override { return "instance_segmentation"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // YOLO26S_SEG_FACTORY_HPP
