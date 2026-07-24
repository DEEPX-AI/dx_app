/**
 * @file realesrgan_x8_factory.hpp
 * @brief RealesrganX8Factory Abstract Factory implementation
 */

#ifndef REALESRGAN_X8_FACTORY_HPP
#define REALESRGAN_X8_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/restoration_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class RealesrganX8Factory : public IRestorationFactory {
public:
    RealesrganX8Factory() = default;

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

    std::string getModelName() const override { return "Realesrgan X8"; }
    std::string getTaskType() const override { return "super_resolution"; }
};

}  // namespace dxapp

#endif  // REALESRGAN_X8_FACTORY_HPP
