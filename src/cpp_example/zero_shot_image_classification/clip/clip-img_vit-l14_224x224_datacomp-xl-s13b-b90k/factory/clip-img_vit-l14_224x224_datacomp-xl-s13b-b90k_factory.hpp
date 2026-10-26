/**
 * @file clip-img_vit-l14_224x224_datacomp-xl-s13b-b90k_factory.hpp
 * @brief ClipFactory Abstract Factory implementation
 */

#ifndef CLIP_IMG_VIT_L14_224X224_DATACOMP_XL_S13B_B90K_FACTORY_HPP
#define CLIP_IMG_VIT_L14_224X224_DATACOMP_XL_S13B_B90K_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_clip_img_vit_l14_224x224_datacomp_xl_s13b_b90k {

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
        return "clip-img_vit-l14_224x224_datacomp-xl-s13b-b90k";
    }
    std::string getTaskType() const override { return "embedding"; }

private:
};

}  // namespace v_clip_img_vit_l14_224x224_datacomp_xl_s13b_b90k
}  // namespace dxapp

#endif  // CLIP_IMG_VIT_L14_224X224_DATACOMP_XL_S13B_B90K_FACTORY_HPP
