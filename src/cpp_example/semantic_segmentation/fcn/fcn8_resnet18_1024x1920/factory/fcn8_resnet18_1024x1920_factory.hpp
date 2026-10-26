/**
 * @file fcn8_resnet18_1024x1920_factory.hpp
 * @brief FcnFactory Abstract Factory implementation
 */

#ifndef FCN8_RESNET18_1024X1920_FACTORY_HPP
#define FCN8_RESNET18_1024X1920_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_fcn8_resnet18_1024x1920 {

class FcnFactory : public ISegmentationFactory {
public:

    FcnFactory() = default;

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

    std::string getModelName() const override {
        return "fcn8_resnet18_1024x1920";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_fcn8_resnet18_1024x1920
}  // namespace dxapp

#endif  // FCN8_RESNET18_1024X1920_FACTORY_HPP
