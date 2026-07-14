/**
 * @file deeplabv3_mobilenetv2_factory.hpp
 * @brief Deeplabv3Mobilenetv2Factory Abstract Factory implementation
 */

#ifndef DEEPLABV3_MOBILENETV2_FACTORY_HPP
#define DEEPLABV3_MOBILENETV2_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Deeplabv3Mobilenetv2Factory : public ISegmentationFactory {
public:
    Deeplabv3Mobilenetv2Factory() = default;

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

    std::string getModelName() const override { return "Deeplabv3 Mobilenetv2"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // DEEPLABV3_MOBILENETV2_FACTORY_HPP
