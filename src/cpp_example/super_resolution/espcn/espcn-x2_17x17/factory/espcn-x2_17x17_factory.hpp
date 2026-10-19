/**
 * @file espcn-x2_17x17_factory.hpp
 * @brief EspcnFactory Abstract Factory implementation
 */

#ifndef ESPCN_X2_17X17_FACTORY_HPP
#define ESPCN_X2_17X17_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/espcn_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/grayscale_preprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_espcn_x2_17x17 {

class EspcnFactory : public IRestorationFactory {
public:

    EspcnFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // default: espcn-x2_17x17
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        // default: espcn-x2_17x17
        return std::make_unique<ESPCNPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override {
        return "espcn-x2_17x17";
    }
    std::string getTaskType() const override { return "super_resolution"; }

private:
};

}  // namespace v_espcn_x2_17x17
}  // namespace dxapp

#endif  // ESPCN_X2_17X17_FACTORY_HPP
