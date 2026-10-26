/**
 * @file eigenplaces-resnet18_512x512_factory.hpp
 * @brief EigenplacesFactory Abstract Factory implementation
 */

#ifndef EIGENPLACES_RESNET18_512X512_FACTORY_HPP
#define EIGENPLACES_RESNET18_512X512_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/gallery_retrieval_postprocessor.hpp"
#include "common/visualizers/retrieval_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_eigenplaces_resnet18_512x512 {

class EigenplacesFactory : public IEmbeddingFactory {
public:

    EigenplacesFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<EmbeddingResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<GalleryRetrievalPostprocessor>(
            input_width, input_height, gallery_, top_k_,
            "eigenplaces-resnet18_512x512");
    }

    VisualizerPtr<EmbeddingResult> createVisualizer() override {
        return std::make_unique<RetrievalVisualizer>(
            title_, max_display_);
    }


    /// ModelConfig is a FLAT parser -- it skips nested objects -- so these keys are
    /// mirrored at the top level of config.json next to the nested `config` block the
    /// Python tree reads. test_zoo_task_categories pins the two copies equal.
    void loadConfig(const dxapp::ModelConfig& config) override {
        gallery_ = config.get<std::string>("gallery", gallery_);
        top_k_ = config.get<int>("top_k", top_k_);
        max_display_ = config.get<int>("max_display", max_display_);
        title_ = config.get<std::string>("title", title_);
    }

    std::string getModelName() const override {
        return "eigenplaces-resnet18_512x512";
    }
    std::string getTaskType() const override { return "visual_place_recognition"; }

private:
    // The gallery the descriptor is ranked against, built on the NPU by
    // scripts/build_gallery_database.py. The .bin is the C++-readable twin of the
    // .npz the Python tree loads; both are written in one call from one array.
    std::string gallery_{"sample/gallery/vpr_eigenplaces-resnet18_512x512.bin"};
    int top_k_{5};
    int max_display_{3};
    std::string title_{"Visual Place Recognition - EigenPlaces ResNet-18"};
};

}  // namespace v_eigenplaces_resnet18_512x512
}  // namespace dxapp

#endif  // EIGENPLACES_RESNET18_512X512_FACTORY_HPP
