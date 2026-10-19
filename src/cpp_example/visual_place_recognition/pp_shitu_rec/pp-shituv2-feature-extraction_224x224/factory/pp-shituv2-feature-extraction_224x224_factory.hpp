/**
 * @file pp-shituv2-feature-extraction_224x224_factory.hpp
 * @brief PpShituRec Abstract Factory implementation
 */

#ifndef PP_SHITUV2_FEATURE_EXTRACTION_224X224_FACTORY_HPP
#define PP_SHITUV2_FEATURE_EXTRACTION_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/gallery_retrieval_postprocessor.hpp"
#include "common/visualizers/retrieval_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

// Carried over from super_resolution/eigenplaces: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what pp_shitu_rec needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_pp_shituv2_feature_extraction_224x224 {

class PpShituRecFactory : public IEmbeddingFactory {
public:

    PpShituRecFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<EmbeddingResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<GalleryRetrievalPostprocessor>(
            input_width, input_height, gallery_, top_k_,
            "pp-shituv2-feature-extraction_224x224");
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
        return "pp-shituv2-feature-extraction_224x224";
    }
    std::string getTaskType() const override { return "visual_place_recognition"; }

private:
    // The gallery the descriptor is ranked against, built on the NPU by
    // scripts/build_gallery_database.py. The .bin is the C++-readable twin of the
    // .npz the Python tree loads; both are written in one call from one array.
    std::string gallery_{"sample/gallery/vpr_pp-shituv2-feature-extraction_224x224.bin"};
    int top_k_{5};
    int max_display_{3};
    std::string title_{"Visual Place Recognition - PP-ShiTuV2 features"};
};

}  // namespace v_pp_shituv2_feature_extraction_224x224
}  // namespace dxapp

#endif  // PP_SHITUV2_FEATURE_EXTRACTION_224X224_FACTORY_HPP
