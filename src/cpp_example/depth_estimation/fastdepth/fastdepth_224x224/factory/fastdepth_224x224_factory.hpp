/**
 * @file fastdepth_224x224_factory.hpp
 * @brief FastDepth_1 Abstract Factory implementation for depth estimation
 * 
 * Uses v3-native depth estimation postprocessor.
 */

#ifndef FASTDEPTH_224X224_FACTORY_HPP
#define FASTDEPTH_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/depth_postprocessor.hpp"
#include "common/visualizers/depth_visualizer.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_fastdepth_224x224 {

class FastdepthFactory : public IDepthEstimationFactory {
public:

    FastdepthFactory() = default;

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

    std::string getModelName() const override {
        return "fastdepth_224x224";
    }
    std::string getTaskType() const override { return "depth_estimation"; }

private:
};

}  // namespace v_fastdepth_224x224
}  // namespace dxapp

#endif  // FASTDEPTH_224X224_FACTORY_HPP
