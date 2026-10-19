/**
 * @file clip-img_vit-b16-quickgelu_224x224_metaclip-fullcc_factory.hpp
 * @brief ClipFactory Abstract Factory implementation
 */

#ifndef CLIP_IMG_VIT_B16_QUICKGELU_224X224_METACLIP_FULLCC_FACTORY_HPP
#define CLIP_IMG_VIT_B16_QUICKGELU_224X224_METACLIP_FULLCC_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_clip_img_vit_b16_quickgelu_224x224_metaclip_fullcc {

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
        return "clip-img_vit-b16-quickgelu_224x224_metaclip-fullcc";
    }
    std::string getTaskType() const override { return "embedding"; }

private:
};

}  // namespace v_clip_img_vit_b16_quickgelu_224x224_metaclip_fullcc
}  // namespace dxapp

#endif  // CLIP_IMG_VIT_B16_QUICKGELU_224X224_METACLIP_FULLCC_FACTORY_HPP
