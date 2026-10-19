/**
 * @file yolo26-x-obb_1024x1024_factory.hpp
 * @brief Yolo26l_obb Abstract Factory implementation
 * 
 * Creates matching components for YOLOv26 OBB (Oriented Bounding Box) detection.
 */

#ifndef YOLO26_X_OBB_1024X1024_FACTORY_HPP
#define YOLO26_X_OBB_1024X1024_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/obb_postprocessor.hpp"
#include "common/visualizers/obb_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolo26_x_obb_1024x1024 {

class Yolo26ObbFactory : public IOBBFactory {
public:

    explicit Yolo26ObbFactory(float score_threshold = 0.3f)
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

    std::string getModelName() const override {
        return "yolo26-x-obb_1024x1024";
    }
    std::string getTaskType() const override { return "obb_detection"; }

private:
    float score_threshold_;
};

}  // namespace v_yolo26_x_obb_1024x1024
}  // namespace dxapp

#endif  // YOLO26_X_OBB_1024X1024_FACTORY_HPP
