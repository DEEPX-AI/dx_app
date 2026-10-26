/**
 * @file rtdetrv2-r18vd-120e_640x640_factory.hpp
 * @brief RT-DETR factory
 *
 * Hand-written, not carried over from a donor: no existing family in this tree
 * consumes this model's output, so there is no factory whose behaviour would be right.
 * RT-DETR is NMS-free, so nms_threshold defaults to 1.0 (off).
 */

#ifndef RTDETRV2_R18VD_120E_640X640_FACTORY_HPP
#define RTDETRV2_R18VD_120E_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/rtdetr_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>
#include <vector>

namespace dxapp {
namespace v_rtdetrv2_r18vd_120e_640x640 {

class RtdetrFactory : public IDetectionFactory {
public:

    RtdetrFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<RTDETRPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            top_k_, layout_, class_names_);
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        top_k_ = config.get<int>("top_k", top_k_);
        layout_ = config.get<std::string>("layout", layout_);
        class_names_ = config.get_string_list("class_names");
    }

    std::string getModelName() const override {
        return "rtdetrv2-r18vd-120e_640x640";
    }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_{0.4f};
    // 1.0 == off. The model is NMS-free; this only merges the
    // same box surviving under two classes.
    float nms_threshold_{1.0f};
    int top_k_{300};
    std::string layout_{"auto"};
    std::vector<std::string> class_names_;
};

}  // namespace v_rtdetrv2_r18vd_120e_640x640
}  // namespace dxapp

#endif  // RTDETRV2_R18VD_120E_640X640_FACTORY_HPP
