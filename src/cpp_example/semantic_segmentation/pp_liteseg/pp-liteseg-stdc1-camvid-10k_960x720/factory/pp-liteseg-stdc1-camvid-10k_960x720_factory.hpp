/**
 * @file pp-liteseg-stdc1-camvid-10k_960x720_factory.hpp
 * @brief PpLiteseg Abstract Factory implementation for semantic segmentation
 * 
 * Uses unified DeepLabv3Postprocessor (NCHW/NHWC, int16/float argmax).
 */

#ifndef PP_LITESEG_STDC1_CAMVID_10K_960X720_FACTORY_HPP
#define PP_LITESEG_STDC1_CAMVID_10K_960X720_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

#include <string>
#include <utility>

// Carried over from semantic_segmentation/bisenet: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what pp_liteseg needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_pp_liteseg_stdc1_camvid_10k_960x720 {

class PpLitesegFactory : public ISegmentationFactory {
public:

    PpLitesegFactory() = default;

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
        return "pp-liteseg-stdc1-camvid-10k_960x720";
    }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
};

}  // namespace v_pp_liteseg_stdc1_camvid_10k_960x720
}  // namespace dxapp

#endif  // PP_LITESEG_STDC1_CAMVID_10K_960X720_FACTORY_HPP
