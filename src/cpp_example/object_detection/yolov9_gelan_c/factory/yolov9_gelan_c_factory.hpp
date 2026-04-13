/**
 * @file yolov9_gelan_c_factory.hpp
 * @brief YOLOv9_gelan_c Abstract Factory implementation
 * 
 * Note: YOLOv8 is anchor-free, no obj_threshold needed
 */

#ifndef YOLOV9_GELAN_C_FACTORY_HPP
#define YOLOV9_GELAN_C_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class YOLOv9_gelan_cFactory : public IDetectionFactory {
public:
    YOLOv9_gelan_cFactory(float score_threshold = 0.3f,
                  float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<YOLOv8Postprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            is_ort_configured
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "YOLOv9_gelan_c"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // YOLOV9_GELAN_C_FACTORY_HPP
