/**
 * @file yolo26n_obb_factory.hpp
 * @brief Yolo26n_obb Abstract Factory implementation
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Creates matching components for YOLOv26 OBB (Oriented Bounding Box) detection.
 */

#ifndef YOLO26N_OBB_FACTORY_HPP
#define YOLO26N_OBB_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/obb_postprocessor.hpp"
#include "common/visualizers/obb_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Yolo26n_obbFactory : public IOBBFactory {
public:
    explicit Yolo26n_obbFactory(float score_threshold = 0.3f)
        : score_threshold_(score_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<OBBResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        return std::make_unique<YOLOv26OBBPostprocessor>(
            input_width, input_height,
            score_threshold_,
            is_ort_configured
        );
    }

    VisualizerPtr<OBBResult> createVisualizer() override {
        return std::make_unique<OBBVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
    }

    std::string getModelName() const override { return "Yolo26n_obb"; }
    std::string getTaskType() const override { return "obb_detection"; }

private:
    float score_threshold_;
};

}  // namespace dxapp

#endif  // YOLO26N_OBB_FACTORY_HPP
