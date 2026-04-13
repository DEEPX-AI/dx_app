/**
 * @file yolov7_w6_wo_decoding_factory.hpp
 * @brief YOLOv7_w6_wo_decoding Abstract Factory implementation
 */

#ifndef YOLOV7_W6_WO_DECODING_FACTORY_HPP
#define YOLOV7_W6_WO_DECODING_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class YOLOv7_w6_wo_decodingFactory : public IDetectionFactory {
public:
    YOLOv7_w6_wo_decodingFactory(float obj_threshold = 0.25f,
                  float score_threshold = 0.25f,
                  float nms_threshold = 0.45f)
        : obj_threshold_(obj_threshold),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<YOLOv7Postprocessor>(
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

    std::string getModelName() const override { return "YOLOv7_w6_wo_decoding"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // YOLOV7_W6_WO_DECODING_FACTORY_HPP
