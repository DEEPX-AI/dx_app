/**
 * @file eigenplaces_resnet50_factory.hpp
 * @brief EigenplacesResnet50Factory Abstract Factory implementation
 */

#ifndef EIGENPLACES_RESNET50_FACTORY_HPP
#define EIGENPLACES_RESNET50_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class EigenplacesResnet50Factory : public IEmbeddingFactory {
public:
    EigenplacesResnet50Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<EmbeddingResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<GenericEmbeddingPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<EmbeddingResult> createVisualizer() override {
        return std::make_unique<EmbeddingVisualizer>();
    }

    std::string getModelName() const override { return "Eigenplaces Resnet50"; }
    std::string getTaskType() const override { return "embedding"; }
};

}  // namespace dxapp

#endif  // EIGENPLACES_RESNET50_FACTORY_HPP
