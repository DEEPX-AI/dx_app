/**
 * @file 3ddfa_v2_mobilnetv1_120x120_factory.hpp
 * @brief 3DDFA v2 MobileNetV1 Abstract Factory for face alignment
 *
 * Uses TDDFA postprocessor with simple resize preprocessor.
 */

#ifndef TDDFA_3DDFA_V2_MOBILNETV1_120X120_FACTORY_HPP
#define TDDFA_3DDFA_V2_MOBILNETV1_120X120_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/tddfa_postprocessor.hpp"
#include "common/visualizers/face_alignment_visualizer.hpp"

namespace dxapp {

class Tddfa3ddfa_v2_mobilnetv1_120x120Factory : public IFaceAlignmentFactory {
public:
    Tddfa3ddfa_v2_mobilnetv1_120x120Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<FaceAlignmentResult> createPostprocessor(
        int input_width, int input_height, bool /*is_ort_configured*/ = false) override {
        return std::make_unique<TDDFAPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<FaceAlignmentResult> createVisualizer() override {
        return std::make_unique<FaceAlignmentVisualizer>();
    }

    std::string getModelName() const override { return "3ddfa_v2_mobilnetv1_120x120"; }
    std::string getTaskType() const override { return "face_alignment"; }
};

}  // namespace dxapp

#endif  // TDDFA_3DDFA_V2_MOBILNETV1_120X120_FACTORY_HPP
