/**
 * @file clip-text_resnet50_77x512_openai_factory.hpp
 * @brief ClipFactory Abstract Factory implementation
 */

#ifndef CLIP_TEXT_RESNET50_77X512_OPENAI_FACTORY_HPP
#define CLIP_TEXT_RESNET50_77X512_OPENAI_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_clip_text_resnet50_77x512_openai {

class ClipFactory : public IEmbeddingFactory {
public:

    ClipFactory() = default;

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

    std::string getModelName() const override {
        return "clip-text_resnet50_77x512_openai";
    }
    std::string getTaskType() const override { return "image_retrieval"; }

private:
};

}  // namespace v_clip_text_resnet50_77x512_openai
}  // namespace dxapp

#endif  // CLIP_TEXT_RESNET50_77X512_OPENAI_FACTORY_HPP
