/**
 * @file casvit_t_seg_fpn_resnet50_factory.hpp
 * @brief CasvitTSegFpnResnet50Factory Abstract Factory implementation
 */

#ifndef CASVIT_T_SEG_FPN_RESNET50_FACTORY_HPP
#define CASVIT_T_SEG_FPN_RESNET50_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class CasvitTSegFpnResnet50Factory : public ISegmentationFactory {
public:
    CasvitTSegFpnResnet50Factory() = default;

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

    std::string getModelName() const override { return "Casvit T Seg Fpn Resnet50"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // CASVIT_T_SEG_FPN_RESNET50_FACTORY_HPP
