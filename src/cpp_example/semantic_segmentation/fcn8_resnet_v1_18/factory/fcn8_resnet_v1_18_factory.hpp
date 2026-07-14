/**
 * @file fcn8_resnet18_factory.hpp
 * @brief Fcn8Resnet18Factory Abstract Factory implementation
 */

#ifndef FCN8_RESNET18_FACTORY_HPP
#define FCN8_RESNET18_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Fcn8Resnet18Factory : public ISegmentationFactory {
public:
    Fcn8Resnet18Factory() = default;

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

    std::string getModelName() const override { return "Fcn8 Resnet18"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // FCN8_RESNET18_FACTORY_HPP
