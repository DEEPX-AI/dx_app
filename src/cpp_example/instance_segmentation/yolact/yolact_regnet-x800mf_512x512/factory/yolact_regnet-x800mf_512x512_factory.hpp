/**
 * @file yolact_regnet-x800mf_512x512_factory.hpp
 * @brief YOLACT RegNetX-800MF Abstract Factory for instance segmentation
 *
 * SSD-based instance segmentation with prototype masks.
 */

#ifndef YOLACT_REGNET_X800MF_512X512_FACTORY_HPP
#define YOLACT_REGNET_X800MF_512X512_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/yolact_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolact_regnet_x800mf_512x512 {

class YolactFactory : public IInstanceSegmentationFactory {
public:

    YolactFactory(float score_threshold = 0.3f,
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
            num_classes_, num_protos_, top_k_,
            class_names_
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
        class_names_ = config.get_string_list("class_names");
        top_k_ = config.get<int>("top_k", top_k_);
    }

    std::string getModelName() const override {
        return "yolact_regnet-x800mf_512x512";
    }
    std::string getTaskType() const override { return "instance_segmentation"; }

private:
    float score_threshold_;
    float nms_threshold_;
    int num_classes_;
    int num_protos_;
    int top_k_;
    std::vector<std::string> class_names_;
};

}  // namespace v_yolact_regnet_x800mf_512x512
}  // namespace dxapp

#endif  // YOLACT_REGNET_X800MF_512X512_FACTORY_HPP
