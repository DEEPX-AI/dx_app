/**
 * @file realesrgan-x4_192x192_factory.hpp
 * @brief RealesrganFactory Abstract Factory implementation
 */

#ifndef REALESRGAN_X4_192X192_FACTORY_HPP
#define REALESRGAN_X4_192X192_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/restoration_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_realesrgan_x4_192x192 {

class RealesrganFactory : public IRestorationFactory {
public:

    RealesrganFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<DnCNNPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override {
        return "realesrgan-x4_192x192";
    }
    std::string getTaskType() const override { return "super_resolution"; }

private:
};

}  // namespace v_realesrgan_x4_192x192
}  // namespace dxapp

#endif  // REALESRGAN_X4_192X192_FACTORY_HPP
