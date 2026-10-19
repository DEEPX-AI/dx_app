/**
 * @file yolov5-s6-pose_640x640_factory.hpp
 * @brief YOLOv5Pose-PPU Abstract Factory implementation for pose estimation
 *
 * YOLOv5Pose PPU uses hardware-accelerated postprocessing for pose estimation.
 */

#ifndef YOLOV5_S6_POSE_640X640_FACTORY_HPP
#define YOLOV5_S6_POSE_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/ppu_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/pose_postprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolov5_s6_pose_640x640 {

class Yolov5PoseFactory : public IPoseFactory {
public:

    // Python's YOLOv5PosePostprocessor defaults: this variant's config.json
    // has an empty "config" object, so they are what the runner and a graph
    // stage use.
    Yolov5PoseFactory(float obj_threshold = 0.25f,
                          float score_threshold = 0.3f,
                          float nms_threshold = 0.45f)
        : obj_threshold_(obj_threshold),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<YOLOv5PosePostprocessor>(
            input_width, input_height,
            obj_threshold_, score_threshold_, nms_threshold_,
            is_ort_configured);
    }

    VisualizerPtr<PoseResult> createVisualizer() override {
        return std::make_unique<PoseVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        obj_threshold_ = config.get<float>("obj_threshold", obj_threshold_);
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "yolov5-s6-pose_640x640";
    }
    std::string getTaskType() const override { return "pose_estimation"; }

private:
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace v_yolov5_s6_pose_640x640
}  // namespace dxapp

#endif  // YOLOV5_S6_POSE_640X640_FACTORY_HPP
