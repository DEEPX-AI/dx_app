/**
 * @file faceattr_resnetv1-18_218x178_factory.hpp
 * @brief Face Attribute ResNet18 Abstract Factory for attribute recognition
 */

#ifndef FACEATTR_RESNETV1_18_218X178_FACTORY_HPP
#define FACEATTR_RESNETV1_18_218X178_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/attribute_postprocessor.hpp"
#include "common/visualizers/attribute_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_faceattr_resnetv1_18_218x178 {

class FaceattrFactory : public IClassificationFactory {
public:

    FaceattrFactory(float threshold = 0.5f)
        : threshold_(threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<ClassificationResult> createPostprocessor(
        int /*input_width*/, int /*input_height*/) override {
        return std::make_unique<AttributePostprocessor>(
            threshold_, AttributePostprocessor::LabelSet::CELEBA_40);
    }

    VisualizerPtr<ClassificationResult> createVisualizer() override {
        return std::make_unique<AttributeVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        threshold_ = config.get<float>("threshold", threshold_);
    }

    std::string getModelName() const override {
        return "faceattr_resnetv1-18_218x178";
    }
    std::string getTaskType() const override { return "attribute_recognition"; }

private:
    float threshold_;
};

}  // namespace v_faceattr_resnetv1_18_218x178
}  // namespace dxapp

#endif  // FACEATTR_RESNETV1_18_218X178_FACTORY_HPP
