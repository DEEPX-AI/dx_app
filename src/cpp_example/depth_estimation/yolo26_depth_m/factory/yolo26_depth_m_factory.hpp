/**
 * @file yolo26_depth_m_factory.hpp
 * @brief Yolo26DepthMFactory Abstract Factory implementation
 *
 * YOLO26-Depth-M monocular depth estimation. The compiled model
 * `yolo26-depth-m_768x768.dxnn` takes a uint8 NHWC input
 * `images` [1,768,768,3] and produces a dense float32 depth map
 * `depth` [1,1,768,768].
 */

#ifndef YOLO26_DEPTH_M_FACTORY_HPP
#define YOLO26_DEPTH_M_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/depth_postprocessor.hpp"
#include "common/visualizers/depth_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Yolo26DepthMFactory : public IDepthEstimationFactory {
public:
    Yolo26DepthMFactory() = default;

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

    std::string getModelName() const override { return "YOLO26-Depth-M"; }
    std::string getTaskType() const override { return "depth_estimation"; }

    // getInputNormalization() is intentionally NOT overridden. The compiled model
    // consumes uint8 NHWC directly (the /255 normalization is folded in at
    // compile time), so the default {apply_mean_std = false} keeps the runner
    // on the raw-uint8 path. Declaring a float spec here would hand the engine
    // a 4x-oversized float32 buffer -> out-of-bounds read / segfault.
};

}  // namespace dxapp

#endif  // YOLO26_DEPTH_M_FACTORY_HPP
