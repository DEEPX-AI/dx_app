/**
 * @file yolact_regnetx_800mf_factory.hpp
 * @brief YOLACT RegNetX-800MF Abstract Factory for instance segmentation
 *
 * SSD-based instance segmentation with prototype masks.
 */

#ifndef YOLACT_REGNETX_800MF_FACTORY_HPP
#define YOLACT_REGNETX_800MF_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/yolact_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class YOLACTFactory : public IInstanceSegmentationFactory {
public:
    YOLACTFactory(float score_threshold = 0.3f,
                  float nms_threshold = 0.5f,
                  int num_classes = 81,
                  int num_protos = 32,
                  int top_k = 200)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold),
          num_classes_(num_classes),
          num_protos_(num_protos), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<InstanceSegmentationResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<YOLACTPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            num_classes_, num_protos_, top_k_
        );
    }

    VisualizerPtr<InstanceSegmentationResult> createVisualizer() override {
        return std::make_unique<InstanceSegmentationVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        num_classes_ = config.get<int>("num_classes", num_classes_);
        num_protos_ = config.get<int>("num_protos", num_protos_);
        top_k_ = config.get<int>("top_k", top_k_);
    }

    std::string getModelName() const override { return "YOLACT-RegNetX-800MF"; }
    std::string getTaskType() const override { return "instance_segmentation"; }

private:
    float score_threshold_;
    float nms_threshold_;
    int num_classes_;
    int num_protos_;
    int top_k_;
};

}  // namespace dxapp

#endif  // YOLACT_REGNETX_800MF_FACTORY_HPP
