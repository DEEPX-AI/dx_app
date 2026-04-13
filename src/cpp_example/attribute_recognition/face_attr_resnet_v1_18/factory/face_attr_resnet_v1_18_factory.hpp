/**
 * @file face_attr_resnet_v1_18_factory.hpp
 * @brief Face Attribute ResNet18 Abstract Factory for attribute recognition
 */

#ifndef FACE_ATTR_RESNET_V1_18_FACTORY_HPP
#define FACE_ATTR_RESNET_V1_18_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/attribute_postprocessor.hpp"
#include "common/visualizers/attribute_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Face_attr_ResNet_v1_18Factory : public IClassificationFactory {
public:
    Face_attr_ResNet_v1_18Factory(float threshold = 0.5f)
        : threshold_(threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<ClassificationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<AttributePostprocessor>(
            threshold_, AttributePostprocessor::LabelSet::CELEBA_40);
    }

    VisualizerPtr<ClassificationResult> createVisualizer() override {
        return std::make_unique<AttributeVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        threshold_ = config.get<float>("threshold", threshold_);
    }

    std::string getModelName() const override { return "Face_attr_ResNet_v1_18"; }
    std::string getTaskType() const override { return "attribute_recognition"; }

private:
    float threshold_;
};

}  // namespace dxapp

#endif  // FACE_ATTR_RESNET_V1_18_FACTORY_HPP
