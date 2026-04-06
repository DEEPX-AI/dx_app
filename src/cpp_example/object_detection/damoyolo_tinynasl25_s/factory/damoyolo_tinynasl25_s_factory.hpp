/**
 * @file damoyolo_tinynasl25_s_factory.hpp
 * @brief DAMO-YOLO Abstract Factory implementation
 * 
 * Uses v3-native DAMO-YOLO postprocessor with split-head format.
 */

#ifndef DAMOYOLO_TINYNASL25_S_FACTORY_HPP
#define DAMOYOLO_TINYNASL25_S_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/damoyolo_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class DamoYOLO_tinynasl25_sFactory : public IDetectionFactory {
public:
    DamoYOLO_tinynasl25_sFactory(float score_threshold = 0.3f,
                    float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<DamoYOLOPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "DAMO-YOLO-TinyNASL25-s"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // DAMOYOLO_TINYNASL25_S_FACTORY_HPP
