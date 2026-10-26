/**
 * @file deeplabv3_mobilenetv2_513x513_nodilation_factory.hpp
 * @brief Deeplabv3Factory Abstract Factory implementation
 */

#ifndef DEEPLABV3_MOBILENETV2_513X513_NODILATION_FACTORY_HPP
#define DEEPLABV3_MOBILENETV2_513X513_NODILATION_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/letterbox_preprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_deeplabv3_mobilenetv2_513x513_nodilation {

class Deeplabv3Factory : public ISegmentationFactory {
public:

    Deeplabv3Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // default: deeplabv3-resnet101_512x512
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) override {
        // default: deeplabv3-resnet101_512x512
        return std::make_unique<DeepLabv3Postprocessor>(input_width, input_height);
    }

    VisualizerPtr<SegmentationResult> createVisualizer() override {
        return std::make_unique<SemanticSegmentationVisualizer>();
    }

    std::string getModelName() const override {
        return "deeplabv3_mobilenetv2_513x513_nodilation";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_deeplabv3_mobilenetv2_513x513_nodilation
}  // namespace dxapp

#endif  // DEEPLABV3_MOBILENETV2_513X513_NODILATION_FACTORY_HPP
