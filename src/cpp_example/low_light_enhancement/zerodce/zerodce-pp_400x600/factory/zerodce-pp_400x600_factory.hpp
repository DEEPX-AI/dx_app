/**
 * @file zerodce-pp_400x600_factory.hpp
 * @brief ZerodceFactory Abstract Factory implementation for ZeroDCE++ image enhancement
 *
 * ZeroDCE++ outputs 4 iterations × 3 channels = 12-channel curve parameter maps.
 * The runner auto-detects the number of iterations from the output tensor.
 */

#ifndef ZERODCE_PP_400X600_FACTORY_HPP
#define ZERODCE_PP_400X600_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/zero_dce_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_zerodce_pp_400x600 {

class ZerodceFactory : public IRestorationFactory {
public:

    ZerodceFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // default: zerodce-pp_400x600
        return std::make_unique<SimpleResizePreprocessor>(
            input_width, input_height, cv::COLOR_BGR2RGB, true);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        // default: zerodce-pp_400x600
        return std::make_unique<ZeroDCEPostprocessor>(
            input_width, input_height, 4  // ZeroDCE++ uses 4 iterations
        );
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override {
        return "zerodce-pp_400x600";
    }
    std::string getTaskType() const override { return "image_enhancement"; }

private:
};

}  // namespace v_zerodce_pp_400x600
}  // namespace dxapp

#endif  // ZERODCE_PP_400X600_FACTORY_HPP
