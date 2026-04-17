/**
 * @file deeplabv3plusmobilenetv2_factory.hpp
 * @brief DeepLabv3plusmobilenetv2 Abstract Factory implementation for semantic segmentation
 */

#ifndef DEEPLABV3PLUSMOBILENETV2_FACTORY_HPP
#define DEEPLABV3PLUSMOBILENETV2_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

namespace dxapp {

class DeepLabv3plusmobilenetv2Factory : public ISegmentationFactory {
public:
    DeepLabv3plusmobilenetv2Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<DeepLabv3Postprocessor>(
            input_width, input_height
        );
    }

    VisualizerPtr<SegmentationResult> createVisualizer() override {
        return std::make_unique<SemanticSegmentationVisualizer>();
    }

    std::string getModelName() const override { return "DeepLabv3plusmobilenetv2"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // DEEPLABV3PLUSMOBILENETV2_FACTORY_HPP
