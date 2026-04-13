/**
 * @file arcface_iresnet50_ms1m_factory.hpp
 * @brief ArcFace MobileFaceNet Abstract Factory implementation for embedding
 * 
 * Uses v3-native embedding postprocessor with simple resize preprocessor.
 */

#ifndef ARCFACE_IRESNET50_MS1M_FACTORY_HPP
#define ARCFACE_IRESNET50_MS1M_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"

namespace dxapp {

class Arcface_iResNet50_ms1mFactory : public IEmbeddingFactory {
public:
    Arcface_iResNet50_ms1mFactory() = default;

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

    std::string getModelName() const override { return "Arcface_iResNet50_ms1m"; }
    std::string getTaskType() const override { return "embedding"; }
};

}  // namespace dxapp

#endif  // ARCFACE_IRESNET50_MS1M_FACTORY_HPP
