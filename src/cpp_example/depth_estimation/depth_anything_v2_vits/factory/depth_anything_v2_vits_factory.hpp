/**
 * @file depth_anything_v2_vits_factory.hpp
 * @brief DepthAnythingV2VitsFactory Abstract Factory implementation
 */

#ifndef DEPTH_ANYTHING_V2_VITS_FACTORY_HPP
#define DEPTH_ANYTHING_V2_VITS_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/depth_postprocessor.hpp"
#include "common/visualizers/depth_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class DepthAnythingV2VitsFactory : public IDepthEstimationFactory {
public:
    DepthAnythingV2VitsFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DepthResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<FastDepthPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<DepthResult> createVisualizer() override {
        return std::make_unique<DepthVisualizer>();
    }

    std::string getModelName() const override { return "Depth Anything V2 Vits"; }
    std::string getTaskType() const override { return "depth_estimation"; }

    // Depth Anything V2 expects float32 input normalized with ImageNet mean/std
    // (RGB order, after /255). Without this the runner would feed a uint8 buffer
    // that is 1/4 the size the model reads -> out-of-bounds read / segfault.
    InputNormalizationParams getInputNormalization() const override {
        return {true, {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
    }
};

}  // namespace dxapp

#endif  // DEPTH_ANYTHING_V2_VITS_FACTORY_HPP
