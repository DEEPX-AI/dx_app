/**
 * @file scdepthv3_factory.hpp
 * @brief Scdepthv3 Abstract Factory implementation for depth estimation
 * 
 * Uses v3-native depth estimation postprocessor.
 */

#ifndef SCDEPTHV3_FACTORY_HPP
#define SCDEPTHV3_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/depth_postprocessor.hpp"
#include "common/visualizers/depth_visualizer.hpp"

namespace dxapp {

class Scdepthv3Factory : public IDepthEstimationFactory {
public:
    Scdepthv3Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DepthResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<FastDepthPostprocessor>(
            input_width, input_height
        );
    }

    VisualizerPtr<DepthResult> createVisualizer() override {
        return std::make_unique<DepthVisualizer>();
    }

    std::string getModelName() const override { return "Scdepthv3"; }
    std::string getTaskType() const override { return "depth_estimation"; }
};

}  // namespace dxapp

#endif  // SCDEPTHV3_FACTORY_HPP
