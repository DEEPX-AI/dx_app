/**
 * @file ulfgfd_rfb_320_factory.hpp
 * @brief Ulfgfd_rfb_320 Abstract Factory implementation for face detection
 */

#ifndef ULFGFD_RFB_320_FACTORY_HPP
#define ULFGFD_RFB_320_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/ulfg_face_postprocessor.hpp"
#include "common/visualizers/face_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Ulfgfd_rfb_320Factory : public IFaceDetectionFactory {
public:
    Ulfgfd_rfb_320Factory(float score_threshold = 0.5f,
                 float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<FaceDetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<ULFGFacePostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_
        );
    }

    VisualizerPtr<FaceDetectionResult> createVisualizer() override {
        return std::make_unique<FaceVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "Ulfgfd_rfb_320"; }
    std::string getTaskType() const override { return "face_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // ULFGFD_RFB_320_FACTORY_HPP
