/**
 * @file zero_dce_pp_factory.hpp
 * @brief ZeroDCEPPFactory Abstract Factory implementation for ZeroDCE++ image enhancement
 *
 * ZeroDCE++ outputs 4 iterations × 3 channels = 12-channel curve parameter maps.
 * The runner auto-detects the number of iterations from the output tensor.
 */

#ifndef ZERO_DCE_PP_FACTORY_HPP
#define ZERO_DCE_PP_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/zero_dce_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class ZeroDCEPPFactory : public IRestorationFactory {
public:
    ZeroDCEPPFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(
            input_width, input_height, cv::COLOR_BGR2RGB, true);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<ZeroDCEPostprocessor>(
            input_width, input_height, 4  // ZeroDCE++ uses 4 iterations
        );
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override { return "Zero-DCE++"; }
    std::string getTaskType() const override { return "image_enhancement"; }
};

}  // namespace dxapp

#endif  // ZERO_DCE_PP_FACTORY_HPP
