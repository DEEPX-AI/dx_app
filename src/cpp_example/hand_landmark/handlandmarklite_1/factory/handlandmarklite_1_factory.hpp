/**
 * @file handlandmarklite_1_factory.hpp
 * @brief Hand Landmark Lite Abstract Factory implementation for hand landmark
 */

#ifndef HANDLANDMARKLITE_1_FACTORY_HPP
#define HANDLANDMARKLITE_1_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/config/model_config.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/hand_landmark_postprocessor.hpp"
#include "common/visualizers/hand_landmark_visualizer.hpp"

namespace dxapp {

class Handlandmarklite_1Factory : public IHandLandmarkFactory {
public:
    Handlandmarklite_1Factory(float confidence_threshold = 0.5f)
        : confidence_threshold_(confidence_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<HandLandmarkResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<HandLandmarkPostprocessor>(
            input_width, input_height, confidence_threshold_);
    }

    VisualizerPtr<HandLandmarkResult> createVisualizer() override {
        return std::make_unique<HandLandmarkVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        confidence_threshold_ = config.get<float>("confidence_threshold", confidence_threshold_);
    }

    std::string getModelName() const override { return "Handlandmarklite_1"; }
    std::string getTaskType() const override { return "hand_landmark"; }

private:
    float confidence_threshold_;
};

}  // namespace dxapp

#endif  // HANDLANDMARKLITE_1_FACTORY_HPP
