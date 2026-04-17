/**
 * @file espcn_x3_factory.hpp
 * @brief ESPCN x4 Abstract Factory implementation for super-resolution
 * 
 * Uses ESPCN-specific postprocessor with Y-channel SR + CbCr color restoration.
 * GrayscaleResizePreprocessor stores source BGR for YCbCr color merge.
 */

#ifndef ESPCN_X3_FACTORY_HPP
#define ESPCN_X3_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/grayscale_preprocessor.hpp"
#include "common/processors/espcn_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class Espcn_x3Factory : public IRestorationFactory {
public:
    Espcn_x3Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // store_source=true: store BGR image for CbCr extraction in postprocessor
        return std::make_unique<GrayscaleResizePreprocessor>(input_width, input_height, true);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<ESPCNPostprocessor>(
            input_width, input_height, 4  // scale_factor=4
        );
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override { return "Espcn_x3"; }
    std::string getTaskType() const override { return "super_resolution"; }
};

}  // namespace dxapp

#endif  // ESPCN_X3_FACTORY_HPP
