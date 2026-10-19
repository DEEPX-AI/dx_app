/**
 * @file YOLOV5Pose_PPU_factory.hpp
 * @brief YOLOv5Pose-PPU Abstract Factory implementation for pose estimation
 *
 * YOLOv5Pose PPU uses hardware-accelerated postprocessing for pose estimation.
 */

#ifndef YOLOV5POSE_PPU_FACTORY_HPP
#define YOLOV5POSE_PPU_FACTORY_HPP

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
namespace v_YOLOV5Pose_PPU {

class Yolov5PoseFactory : public IPoseFactory {
public:

    Yolov5PoseFactory(float obj_threshold = 0.5f,
                          float score_threshold = 0.5f,
                          float nms_threshold = 0.45f)
        : obj_threshold_(obj_threshold),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // default: YOLOV5Pose_PPU
        return std::make_unique<YOLOv5PosePPUPostprocessor>(
            input_width, input_height,
            obj_threshold_, score_threshold_, nms_threshold_,
            is_ort_configured
        );
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
        return "YOLOV5Pose_PPU";
    }
    std::string getTaskType() const override { return "pose_estimation"; }

private:
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace v_YOLOV5Pose_PPU
}  // namespace dxapp

#endif  // YOLOV5POSE_PPU_FACTORY_HPP
