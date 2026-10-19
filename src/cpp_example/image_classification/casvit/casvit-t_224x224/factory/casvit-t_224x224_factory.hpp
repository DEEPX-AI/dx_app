/**
 * @file casvit-t_224x224_factory.hpp
 * @brief ArcFace MobileFaceNet Abstract Factory implementation for embedding
 * 
 * Uses v3-native embedding postprocessor with simple resize preprocessor.
 */

#ifndef CASVIT_T_224X224_FACTORY_HPP
#define CASVIT_T_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_casvit_t_224x224 {

class CasvitFactory : public IEmbeddingFactory {
public:

    CasvitFactory() = default;

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
        return "casvit-t_224x224";
    }
    std::string getTaskType() const override { return "reid"; }

private:
};

}  // namespace v_casvit_t_224x224
}  // namespace dxapp

#endif  // CASVIT_T_224X224_FACTORY_HPP
