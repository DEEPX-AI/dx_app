/**
 * @file bisenetv1_1024x2048_factory.hpp
 * @brief BiseNetV1 Abstract Factory implementation for semantic segmentation
 * 
 * Uses unified DeepLabv3Postprocessor (NCHW/NHWC, int16/float argmax).
 */

#ifndef BISENETV1_1024X2048_FACTORY_HPP
#define BISENETV1_1024X2048_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_bisenetv1_1024x2048 {

class BisenetFactory : public ISegmentationFactory {
public:

    BisenetFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
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

    std::string getModelName() const override {
        return "bisenetv1_1024x2048";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_bisenetv1_1024x2048
}  // namespace dxapp

#endif  // BISENETV1_1024X2048_FACTORY_HPP
