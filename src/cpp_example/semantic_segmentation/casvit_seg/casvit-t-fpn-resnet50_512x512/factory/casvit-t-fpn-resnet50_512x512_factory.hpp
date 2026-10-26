/**
 * @file casvit-t-fpn-resnet50_512x512_factory.hpp
 * @brief CasvitSegFactory Abstract Factory implementation
 */

#ifndef CASVIT_T_FPN_RESNET50_512X512_FACTORY_HPP
#define CASVIT_T_FPN_RESNET50_512X512_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_casvit_t_fpn_resnet50_512x512 {

class CasvitSegFactory : public ISegmentationFactory {
public:

    CasvitSegFactory() = default;

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
        return "casvit-t-fpn-resnet50_512x512";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_casvit_t_fpn_resnet50_512x512
}  // namespace dxapp

#endif  // CASVIT_T_FPN_RESNET50_512X512_FACTORY_HPP
