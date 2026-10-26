/**
 * @file yolo26-l-pose_640x640_factory.hpp
 * @brief Yolo26l_pose Abstract Factory implementation
 * 
 * Uses v3-native YOLOv8Pose postprocessor (anchor-free, transposed output).
 */

#ifndef YOLO26_L_POSE_640X640_FACTORY_HPP
#define YOLO26_L_POSE_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/pose_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolo26_l_pose_640x640 {

class Yolo26PoseFactory : public IPoseFactory {
public:

    Yolo26PoseFactory(float score_threshold = 0.3f,
                      float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<YOLOv8PosePostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_
        );
    }

    VisualizerPtr<PoseResult> createVisualizer() override {
        return std::make_unique<PoseVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "yolo26-l-pose_640x640";
    }
    std::string getTaskType() const override { return "pose_estimation"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace v_yolo26_l_pose_640x640
}  // namespace dxapp

#endif  // YOLO26_L_POSE_640X640_FACTORY_HPP
