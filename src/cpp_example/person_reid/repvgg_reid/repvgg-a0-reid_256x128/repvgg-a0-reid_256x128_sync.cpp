/**
 * @file repvgg-a0-reid_256x128
 * @brief repvgg_reid sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/repvgg-a0-reid_256x128_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_repvgg_a0_reid_256x128::RepvggReidFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::v_repvgg_a0_reid_256x128::RepvggReidFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
