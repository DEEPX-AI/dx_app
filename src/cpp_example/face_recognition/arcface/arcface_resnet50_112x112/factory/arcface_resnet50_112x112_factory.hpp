/**
 * @file arcface_resnet50_112x112_factory.hpp
 * @brief ArcFace MobileFaceNet Abstract Factory implementation for embedding
 * 
 * Uses v3-native embedding postprocessor with simple resize preprocessor.
 */

#ifndef ARCFACE_RESNET50_112X112_FACTORY_HPP
#define ARCFACE_RESNET50_112X112_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_arcface_resnet50_112x112 {

class ArcfaceFactory : public IEmbeddingFactory {
public:

    ArcfaceFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<EmbeddingResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<GenericEmbeddingPostprocessor>(
            input_width, input_height, true /* L2 normalize */
        );
    }

    VisualizerPtr<EmbeddingResult> createVisualizer() override {
        return std::make_unique<EmbeddingVisualizer>();
    }

    std::string getModelName() const override {
        return "arcface_resnet50_112x112";
    }
    std::string getTaskType() const override { return "embedding"; }

private:
};

}  // namespace v_arcface_resnet50_112x112
}  // namespace dxapp

#endif  // ARCFACE_RESNET50_112X112_FACTORY_HPP
