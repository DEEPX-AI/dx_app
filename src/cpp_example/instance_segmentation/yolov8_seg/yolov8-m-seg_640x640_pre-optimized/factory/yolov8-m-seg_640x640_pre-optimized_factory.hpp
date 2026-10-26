/**
 * @file yolov8-m-seg_640x640_pre-optimized_factory.hpp
 * @brief Yolov8SegFactory Abstract Factory implementation
 */

#ifndef YOLOV8_M_SEG_640X640_PRE_OPTIMIZED_FACTORY_HPP
#define YOLOV8_M_SEG_640X640_PRE_OPTIMIZED_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolov8_m_seg_640x640_pre_optimized {

class Yolov8SegFactory : public IInstanceSegmentationFactory {
public:

    Yolov8SegFactory(float score_threshold = 0.5f, float nms_threshold = 0.65f)
        : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<InstanceSegmentationResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // default: yolov8-l-seg_640x640
        return std::make_unique<YOLOv8SegPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            is_ort_configured);
    }

    VisualizerPtr<InstanceSegmentationResult> createVisualizer() override {
        return std::make_unique<InstanceSegmentationVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "yolov8-m-seg_640x640_pre-optimized";
    }
    std::string getTaskType() const override { return "instance_segmentation"; }

private:
    std::vector<std::string> class_names_;  // from a sibling variant in this family
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace v_yolov8_m_seg_640x640_pre_optimized
}  // namespace dxapp

#endif  // YOLOV8_M_SEG_640X640_PRE_OPTIMIZED_FACTORY_HPP
