/**
 * @file nanodet_repvgga12_factory.hpp
 * @brief NanoDet_repvgga12 Abstract Factory implementation
 * 
 * YOLOX-decoded format (raw boxes needing grid decode, post-sigmoid scores).
 */

#ifndef NANODET_REPVGGA12_FACTORY_HPP
#define NANODET_REPVGGA12_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class NanoDet_repvgga12Factory : public IDetectionFactory {
public:
    NanoDet_repvgga12Factory(float obj_threshold = 0.25f,
                   float score_threshold = 0.35f,
                   float nms_threshold = 0.6f)
        : obj_threshold_(obj_threshold),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<YOLOXPostprocessor>(
            input_width, input_height,
            obj_threshold_, score_threshold_, nms_threshold_,
            is_ort_configured
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        obj_threshold_ = config.get<float>("obj_threshold", obj_threshold_);
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "NanoDet_repvgga12"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // NANODET_REPVGGA12_FACTORY_HPP
