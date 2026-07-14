/**
 * @file espcn_x2_factory.hpp
 * @brief EspcnX2Factory Abstract Factory implementation
 */

#ifndef ESPCN_X2_FACTORY_HPP
#define ESPCN_X2_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/espcn_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class EspcnX2Factory : public IRestorationFactory {
public:
    EspcnX2Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<ESPCNPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override { return "Espcn X2"; }
    std::string getTaskType() const override { return "super_resolution"; }
};

}  // namespace dxapp

#endif  // ESPCN_X2_FACTORY_HPP
