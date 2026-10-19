/**
 * @file ddrnet23-slim_1024x2048_factory.hpp
 * @brief Ddrnet Abstract Factory implementation for semantic segmentation
 * 
 * Uses unified DeepLabv3Postprocessor (NCHW/NHWC, int16/float argmax).
 */

#ifndef DDRNET23_SLIM_1024X2048_FACTORY_HPP
#define DDRNET23_SLIM_1024X2048_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

#include <string>
#include <utility>

// Carried over from semantic_segmentation/bisenet: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what ddrnet needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_ddrnet23_slim_1024x2048 {

class DdrnetFactory : public ISegmentationFactory {
public:

    DdrnetFactory() = default;

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
        return "ddrnet23-slim_1024x2048";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_ddrnet23_slim_1024x2048
}  // namespace dxapp

#endif  // DDRNET23_SLIM_1024X2048_FACTORY_HPP
