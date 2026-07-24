/**
 * @file yolov7_lite_t_face_factory.hpp
 * @brief Yolov7LiteTFaceFactory Abstract Factory implementation
 */

#ifndef YOLOV7_LITE_T_FACE_FACTORY_HPP
#define YOLOV7_LITE_T_FACE_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/face_postprocessor.hpp"
#include "common/visualizers/face_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Yolov7LiteTFaceFactory : public IFaceDetectionFactory {
public:
    Yolov7LiteTFaceFactory(float score_threshold = 0.5f, float nms_threshold = 0.4f)
        : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<FaceDetectionResult> createPostprocessor(int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<YOLOv5FacePostprocessor>(
            input_width, input_height, score_threshold_, nms_threshold_);
    }

    VisualizerPtr<FaceDetectionResult> createVisualizer() override {
        return std::make_unique<FaceVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "Yolov7 Lite T Face"; }
    std::string getTaskType() const override { return "face_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // YOLOV7_LITE_T_FACE_FACTORY_HPP
