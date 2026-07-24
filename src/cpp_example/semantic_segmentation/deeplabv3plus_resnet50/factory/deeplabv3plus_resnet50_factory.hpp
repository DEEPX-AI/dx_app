/**
 * @file deeplabv3plus_resnet50_factory.hpp
 * @brief Deeplabv3plusResnet50Factory Abstract Factory implementation
 */

#ifndef DEEPLABV3PLUS_RESNET50_FACTORY_HPP
#define DEEPLABV3PLUS_RESNET50_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Deeplabv3plusResnet50Factory : public ISegmentationFactory {
public:
    Deeplabv3plusResnet50Factory() = default;

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

    std::string getModelName() const override { return "Deeplabv3Plus Resnet50"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // DEEPLABV3PLUS_RESNET50_FACTORY_HPP
