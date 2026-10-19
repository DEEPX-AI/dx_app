/**
 * @file eigenplaces-resnet18_512x512
 * @brief eigenplaces sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/eigenplaces-resnet18_512x512_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_eigenplaces_resnet18_512x512::EigenplacesFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::v_eigenplaces_resnet18_512x512::EigenplacesFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
