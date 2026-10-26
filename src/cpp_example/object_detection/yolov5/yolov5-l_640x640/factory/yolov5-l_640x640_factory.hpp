/**
 * @file yolov5-l_640x640_factory.hpp
 * @brief YOLOv5l6 Abstract Factory implementation
 * 
 */

#ifndef YOLOV5_L_640X640_FACTORY_HPP
#define YOLOV5_L_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolov5_l_640x640 {

class Yolov5Factory : public IDetectionFactory {
public:

    Yolov5Factory(float obj_threshold = 0.25f,
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
        // default: yolov5-l6_1280x1280
        return std::make_unique<YOLOv5Postprocessor>(
            input_width, input_height,
            obj_threshold_, score_threshold_, nms_threshold_,
            is_ort_configured,
            num_classes_,
            class_names_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        obj_threshold_ = config.get<float>("obj_threshold", obj_threshold_);
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        class_names_ = config.get_string_list("class_names");
        num_classes_ = config.get<int>("num_classes", num_classes_);
    }

    std::string getModelName() const override {
        return "yolov5-l_640x640";
    }
    std::string getTaskType() const override { return "object_detection"; }

private:
    int max_nms_candidates_{0};  // from a sibling variant in this family
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
    int num_classes_{80};
    std::vector<std::string> class_names_;
};

}  // namespace v_yolov5_l_640x640
}  // namespace dxapp

#endif  // YOLOV5_L_640X640_FACTORY_HPP
