/**
 * @file deepmar_resnet50_factory.hpp
 * @brief DeepMAR-ResNet50 Abstract Factory for attribute recognition
 */

#ifndef DEEPMAR_RESNET50_FACTORY_HPP
#define DEEPMAR_RESNET50_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/attribute_postprocessor.hpp"
#include "common/visualizers/attribute_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Deepmar_ResNet50Factory : public IClassificationFactory {
public:
    Deepmar_ResNet50Factory(float threshold = 0.5f)
        : threshold_(threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<ClassificationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<AttributePostprocessor>(
            threshold_, AttributePostprocessor::LabelSet::PETA_35);
    }

    VisualizerPtr<ClassificationResult> createVisualizer() override {
        return std::make_unique<AttributeVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        threshold_ = config.get<float>("threshold", threshold_);
    }

    std::string getModelName() const override { return "Deepmar_ResNet50"; }
    std::string getTaskType() const override { return "attribute_recognition"; }

private:
    float threshold_;
};

}  // namespace dxapp

#endif  // DEEPMAR_RESNET50_FACTORY_HPP
