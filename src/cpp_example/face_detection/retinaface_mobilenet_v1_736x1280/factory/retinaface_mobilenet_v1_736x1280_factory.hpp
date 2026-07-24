/**
 * @file retinaface_mobilenetv1_736_factory.hpp
 * @brief RetinafaceMobilenetv1736Factory Abstract Factory implementation
 *
 * RetinaFace MobileNetV1 736x1280 model:
 *   - Input: float32 BGR, mean subtraction [104, 117, 123], NHWC layout
 *   - Output: 9 NHWC feature-map tensors (3 strides × bbox/cls/lmk)
 */

#ifndef RETINAFACE_MOBILENETV1_736_FACTORY_HPP
#define RETINAFACE_MOBILENETV1_736_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/retinaface_postprocessor.hpp"
#include "common/visualizers/face_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class RetinafaceMobilenetv1736Factory : public IFaceDetectionFactory {
public:
    RetinafaceMobilenetv1736Factory(float score_threshold = 0.5f, float nms_threshold = 0.4f)
        : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // BGR float32 with per-channel mean subtraction [104, 117, 123]
        // color_conversion=-1: keep BGR (no RGB conversion)
        return std::make_unique<SimpleResizePreprocessor>(
            input_width, input_height,
            -1,    // keep BGR, no color conversion
            false, // no store_source
            std::array<float, 3>{104.f, 117.f, 123.f},
            true   // output float32
        );
    }

    PostprocessorPtr<FaceDetectionResult> createPostprocessor(int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<RetinaFacePostprocessor>(
            input_width, input_height, score_threshold_, nms_threshold_);
    }

    VisualizerPtr<FaceDetectionResult> createVisualizer() override {
        return std::make_unique<FaceVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "Retinaface Mobilenetv1 736"; }
    std::string getTaskType() const override { return "face_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // RETINAFACE_MOBILENETV1_736_FACTORY_HPP
