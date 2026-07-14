/**
 * @file deeplabv3plus_resnet101_factory.hpp
 * @brief Deeplabv3plusResnet101Factory Abstract Factory implementation
 */

#ifndef DEEPLABV3PLUS_RESNET101_FACTORY_HPP
#define DEEPLABV3PLUS_RESNET101_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Deeplabv3plusResnet101Factory : public ISegmentationFactory {
public:
    Deeplabv3plusResnet101Factory() = default;

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

    std::string getModelName() const override { return "Deeplabv3Plus Resnet101"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // DEEPLABV3PLUS_RESNET101_FACTORY_HPP
