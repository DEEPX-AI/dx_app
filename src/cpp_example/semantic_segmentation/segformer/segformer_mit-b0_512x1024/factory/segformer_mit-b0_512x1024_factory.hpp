/**
 * @file segformer_mit-b0_512x1024_factory.hpp
 * @brief SegFormer-B0 Abstract Factory for semantic segmentation
 *
 * SegFormer outputs argmax-style segmentation maps, reuses DeepLabv3Postprocessor.
 */

#ifndef SEGFORMER_MIT_B0_512X1024_FACTORY_HPP
#define SEGFORMER_MIT_B0_512X1024_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_segformer_mit_b0_512x1024 {

class SegformerFactory : public ISegmentationFactory {
public:

    SegformerFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<DeepLabv3Postprocessor>(
            input_width, input_height, true  // upsample_to_input for smooth boundaries
        );
    }

    VisualizerPtr<SegmentationResult> createVisualizer() override {
        return std::make_unique<SemanticSegmentationVisualizer>();
    }

    std::string getModelName() const override {
        return "segformer_mit-b0_512x1024";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_segformer_mit_b0_512x1024
}  // namespace dxapp

#endif  // SEGFORMER_MIT_B0_512X1024_FACTORY_HPP
