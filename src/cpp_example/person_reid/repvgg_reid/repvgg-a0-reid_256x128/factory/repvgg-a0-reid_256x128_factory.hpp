/**
 * @file repvgg-a0-reid_256x128_factory.hpp
 * @brief RepvggReid MobileFaceNet Abstract Factory implementation for embedding
 * 
 * Uses v3-native embedding postprocessor with simple resize preprocessor.
 */

#ifndef REPVGG_A0_REID_256X128_FACTORY_HPP
#define REPVGG_A0_REID_256X128_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/gallery_retrieval_postprocessor.hpp"
#include "common/visualizers/retrieval_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

// Carried over from image_classification/casvit: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what repvgg_reid needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_repvgg_a0_reid_256x128 {

class RepvggReidFactory : public IEmbeddingFactory {
public:

    RepvggReidFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<EmbeddingResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<GalleryRetrievalPostprocessor>(
            input_width, input_height, gallery_, top_k_,
            "repvgg-a0-reid_256x128");
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
        return "repvgg-a0-reid_256x128";
    }
    std::string getTaskType() const override { return "person_reid"; }

private:
    // The gallery the descriptor is ranked against, built on the NPU by
    // scripts/build_gallery_database.py. The .bin is the C++-readable twin of the
    // .npz the Python tree loads; both are written in one call from one array.
    std::string gallery_{"sample/gallery/reid_repvgg-a0-reid_256x128.bin"};
    int top_k_{5};
    int max_display_{3};
    std::string title_{"Person ReID - RepVGG-A0"};
};

}  // namespace v_repvgg_a0_reid_256x128
}  // namespace dxapp

#endif  // REPVGG_A0_REID_256X128_FACTORY_HPP
