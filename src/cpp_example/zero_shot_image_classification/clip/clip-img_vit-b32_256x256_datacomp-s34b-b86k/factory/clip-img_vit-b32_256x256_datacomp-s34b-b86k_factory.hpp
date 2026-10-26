/**
 * @file clip-img_vit-b32_256x256_datacomp-s34b-b86k_factory.hpp
 * @brief ClipFactory Abstract Factory implementation
 */

#ifndef CLIP_IMG_VIT_B32_256X256_DATACOMP_S34B_B86K_FACTORY_HPP
#define CLIP_IMG_VIT_B32_256X256_DATACOMP_S34B_B86K_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_clip_img_vit_b32_256x256_datacomp_s34b_b86k {

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
        return "clip-img_vit-b32_256x256_datacomp-s34b-b86k";
    }
    std::string getTaskType() const override { return "embedding"; }

private:
};

}  // namespace v_clip_img_vit_b32_256x256_datacomp_s34b_b86k
}  // namespace dxapp

#endif  // CLIP_IMG_VIT_B32_256X256_DATACOMP_S34B_B86K_FACTORY_HPP
