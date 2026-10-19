/**
 * @file swin-b_224x224_factory.hpp
 * @brief Swin Abstract Factory implementation
 */

#ifndef SWIN_B_224X224_FACTORY_HPP
#define SWIN_B_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/letterbox_preprocessor.hpp"

#include <string>
#include <utility>

// Carried over from image_classification/vit: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what swin needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_swin_b_224x224 {

class SwinFactory : public IClassificationFactory {
public:

    SwinFactory(int num_classes = 1000, int top_k = 5)
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
        return "swin-b_224x224";
    }
    std::string getTaskType() const override { return "image_classification"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace v_swin_b_224x224
}  // namespace dxapp

#endif  // SWIN_B_224X224_FACTORY_HPP
