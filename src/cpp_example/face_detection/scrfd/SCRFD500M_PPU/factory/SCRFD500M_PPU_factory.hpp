/**
 * @file SCRFD500M_PPU_factory.hpp
 * @brief SCRFD500M-PPU Abstract Factory implementation for face detection
 *
 * SCRFD PPU uses hardware-accelerated postprocessing for face detection.
 */

#ifndef SCRFD500M_PPU_FACTORY_HPP
#define SCRFD500M_PPU_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/ppu_postprocessor.hpp"
#include "common/visualizers/face_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/face_postprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_SCRFD500M_PPU {

class ScrfdFactory : public IFaceDetectionFactory {
public:

    ScrfdFactory(float score_threshold = 0.5f,
                         float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<FaceDetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // default: SCRFD500M_PPU
        return std::make_unique<SCRFDPPUPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            is_ort_configured
        );
    }

    VisualizerPtr<FaceDetectionResult> createVisualizer() override {
        return std::make_unique<FaceVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "SCRFD500M_PPU";
    }
    std::string getTaskType() const override { return "face_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace v_SCRFD500M_PPU
}  // namespace dxapp

#endif  // SCRFD500M_PPU_FACTORY_HPP
