/**
 * @file pidnet_s_factory.hpp
 * @brief PidnetSFactory Abstract Factory implementation
 */

#ifndef PIDNET_S_FACTORY_HPP
#define PIDNET_S_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class PidnetSFactory : public ISegmentationFactory {
public:
    PidnetSFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<DeepLabv3Postprocessor>(input_width, input_height);
    }

    VisualizerPtr<SegmentationResult> createVisualizer() override {
        return std::make_unique<SemanticSegmentationVisualizer>();
    }

    std::string getModelName() const override { return "Pidnet S"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // PIDNET_S_FACTORY_HPP
