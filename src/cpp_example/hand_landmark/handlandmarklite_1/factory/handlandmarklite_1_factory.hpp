/**
 * @file handlandmarklite_1_factory.hpp
 * @brief Hand Landmark Lite Abstract Factory implementation for hand landmark
 * 
 * Part of DX-APP v3.0.0 refactoring.
 */

#ifndef HANDLANDMARKLITE_1_FACTORY_HPP
#define HANDLANDMARKLITE_1_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/hand_landmark_postprocessor.hpp"
#include "common/visualizers/hand_landmark_visualizer.hpp"

namespace dxapp {

class Handlandmarklite_1Factory : public IHandLandmarkFactory {
public:
    Handlandmarklite_1Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<HandLandmarkResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<HandLandmarkPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<HandLandmarkResult> createVisualizer() override {
        return std::make_unique<HandLandmarkVisualizer>();
    }

    std::string getModelName() const override { return "Handlandmarklite_1"; }
    std::string getTaskType() const override { return "hand_landmark"; }
};

}  // namespace dxapp

#endif  // HANDLANDMARKLITE_1_FACTORY_HPP
