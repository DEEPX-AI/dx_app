/**
 * @file segformer_b0_512x1024_h_factory.hpp
 * @brief Segformer_b0_512x1024_h Abstract Factory for semantic segmentation
 *
 * SegFormer outputs argmax-style segmentation maps, reuses DeepLabv3Postprocessor.
 */

#ifndef SEGFORMER_B0_512X1024_h_FACTORY_HPP
#define SEGFORMER_B0_512X1024_h_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

namespace dxapp {

class Segformer_b0_512x1024_hFactory : public ISegmentationFactory {
public:
    Segformer_b0_512x1024_hFactory() = default;

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

    std::string getModelName() const override { return "Segformer_b0_512x1024_h"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // SEGFORMER_B0_512X1024_h_FACTORY_HPP
