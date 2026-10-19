/**
 * @file mask-rtdetr-hgnetv2-h-6x_640x640_factory.hpp
 * @brief mask-RT-DETR factory
 *
 * Hand-written, not carried over from a donor: no existing family in this tree
 * consumes this model's output, so there is no factory whose behaviour would be right.
 * Same NMS-free query decoding as RT-DETR, plus one mask map per query.
 */

#ifndef MASK_RTDETR_HGNETV2_H_6X_640X640_FACTORY_HPP
#define MASK_RTDETR_HGNETV2_H_6X_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/rtdetr_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>
#include <vector>

namespace dxapp {
namespace v_mask_rtdetr_hgnetv2_h_6x_640x640 {

class MaskRtdetrFactory : public IInstanceSegmentationFactory {
public:

    MaskRtdetrFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<InstanceSegmentationResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<MaskRTDETRPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            top_k_, mask_threshold_, layout_, class_names_);
    }

    VisualizerPtr<InstanceSegmentationResult> createVisualizer() override {
        return std::make_unique<InstanceSegmentationVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        top_k_ = config.get<int>("top_k", top_k_);
        mask_threshold_ = config.get<float>("mask_threshold", mask_threshold_);
        layout_ = config.get<std::string>("layout", layout_);
        class_names_ = config.get_string_list("class_names");
    }

    std::string getModelName() const override {
        return "mask-rtdetr-hgnetv2-h-6x_640x640";
    }
    std::string getTaskType() const override { return "instance_segmentation"; }

private:
    float score_threshold_{0.4f};
    float nms_threshold_{1.0f};
    int top_k_{300};
    float mask_threshold_{0.5f};
    std::string layout_{"auto"};
    std::vector<std::string> class_names_;
};

}  // namespace v_mask_rtdetr_hgnetv2_h_6x_640x640
}  // namespace dxapp

#endif  // MASK_RTDETR_HGNETV2_H_6X_640X640_FACTORY_HPP
