/**
 * @file vit-l-p16_512x512_swag_factory.hpp
 * @brief VitFactory Abstract Factory implementation
 */

#ifndef VIT_L_P16_512X512_SWAG_FACTORY_HPP
#define VIT_L_P16_512X512_SWAG_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/letterbox_preprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_vit_l_p16_512x512_swag {

class VitFactory : public IClassificationFactory {
public:

    VitFactory(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // default: vit-b-p16_224x224
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<ClassificationResult> createPostprocessor(
        int /*input_width*/, int /*input_height*/) override {
        return std::make_unique<EfficientNetPostprocessor>(num_classes_, top_k_);
    }

    VisualizerPtr<ClassificationResult> createVisualizer() override {
        return std::make_unique<ClassificationResultVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        num_classes_ = config.get<int>("num_classes", num_classes_);
        top_k_ = config.get<int>("top_k", top_k_);
    }

    std::string getModelName() const override {
        return "vit-l-p16_512x512_swag";
    }
    std::string getTaskType() const override { return "classification"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace v_vit_l_p16_512x512_swag
}  // namespace dxapp

#endif  // VIT_L_P16_512X512_SWAG_FACTORY_HPP
