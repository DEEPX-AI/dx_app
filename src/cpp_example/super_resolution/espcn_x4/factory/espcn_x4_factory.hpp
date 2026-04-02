/**
 * @file espcn_x4_factory.hpp
 * @brief ESPCN x4 Abstract Factory implementation for super-resolution
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses ESPCN-specific postprocessor with Y-channel SR + CbCr color restoration.
 * GrayscaleResizePreprocessor stores source BGR for YCbCr color merge.
 */

#ifndef ESPCN_X4_FACTORY_HPP
#define ESPCN_X4_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/grayscale_preprocessor.hpp"
#include "common/processors/espcn_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class ESPCNX4Factory : public IRestorationFactory {
public:
    ESPCNX4Factory() = default;

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

    std::string getModelName() const override { return "ESPCN-x4"; }
    std::string getTaskType() const override { return "super_resolution"; }
};

}  // namespace dxapp

#endif  // ESPCN_X4_FACTORY_HPP
