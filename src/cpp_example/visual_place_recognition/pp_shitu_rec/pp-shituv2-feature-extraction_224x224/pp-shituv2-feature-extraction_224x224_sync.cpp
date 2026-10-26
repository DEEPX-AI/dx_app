/**
 * @file pp-shituv2-feature-extraction_224x224
 * @brief pp_shitu_rec sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/pp-shituv2-feature-extraction_224x224_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_pp_shituv2_feature_extraction_224x224::PpShituRecFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::v_pp_shituv2_feature_extraction_224x224::PpShituRecFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
