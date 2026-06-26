/**
 * @file yolov4_leaky_512_512_factory.hpp
 * @brief YOLOv4_leaky_512_512 Abstract Factory implementation
 * 
 */

#ifndef YOLOV4_LEAKY_512_512_FACTORY_HPP
#define YOLOV4_LEAKY_512_512_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/efficientdet_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class YOLOv4_leaky_512_512Factory : public IDetectionFactory {
public:
    YOLOv4_leaky_512_512Factory(float score_threshold = 0.25f,
                  float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<EfficientDetPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            num_classes_,
            class_names_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        class_names_ = config.get_string_list("class_names");
        num_classes_ = config.get<int>("num_classes", num_classes_);
    }

    std::string getModelName() const override { return "YOLOv4_leaky_512_512"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
    int num_classes_{80};
    std::vector<std::string> class_names_;
};

}  // namespace dxapp

#endif  // YOLOV4_LEAKY_512_512_FACTORY_HPP
