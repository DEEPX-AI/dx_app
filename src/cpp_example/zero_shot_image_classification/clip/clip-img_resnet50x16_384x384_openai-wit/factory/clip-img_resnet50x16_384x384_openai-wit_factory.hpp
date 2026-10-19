/**
 * @file clip-img_resnet50x16_384x384_openai-wit_factory.hpp
 * @brief ClipFactory Abstract Factory implementation
 */

#ifndef CLIP_IMG_RESNET50X16_384X384_OPENAI_WIT_FACTORY_HPP
#define CLIP_IMG_RESNET50X16_384X384_OPENAI_WIT_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_clip_img_resnet50x16_384x384_openai_wit {

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
        return "clip-img_resnet50x16_384x384_openai-wit";
    }
    std::string getTaskType() const override { return "embedding"; }

private:
};

}  // namespace v_clip_img_resnet50x16_384x384_openai_wit
}  // namespace dxapp

#endif  // CLIP_IMG_RESNET50X16_384X384_OPENAI_WIT_FACTORY_HPP
