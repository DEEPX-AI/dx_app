/**
 * @file clip_rn50x16_factory.hpp
 * @brief ClipRn50x16Factory Abstract Factory implementation
 */

#ifndef CLIP_RN50X16_FACTORY_HPP
#define CLIP_RN50X16_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class ClipRn50x16Factory : public IEmbeddingFactory {
public:
    ClipRn50x16Factory() = default;

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

    std::string getModelName() const override { return "Clip Rn50X16"; }
    std::string getTaskType() const override { return "embedding"; }
};

}  // namespace dxapp

#endif  // CLIP_RN50X16_FACTORY_HPP
