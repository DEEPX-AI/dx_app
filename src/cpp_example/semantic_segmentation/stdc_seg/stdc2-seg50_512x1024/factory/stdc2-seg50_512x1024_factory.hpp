/**
 * @file stdc2-seg50_512x1024_factory.hpp
 * @brief StdcSegFactory Abstract Factory implementation
 */

#ifndef STDC2_SEG50_512X1024_FACTORY_HPP
#define STDC2_SEG50_512X1024_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_stdc2_seg50_512x1024 {

class StdcSegFactory : public ISegmentationFactory {
public:

    StdcSegFactory() = default;

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
        return "stdc2-seg50_512x1024";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_stdc2_seg50_512x1024
}  // namespace dxapp

#endif  // STDC2_SEG50_512X1024_FACTORY_HPP
