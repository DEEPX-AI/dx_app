/**
 * @file eigenplaces-resnet50_512x512
 * @brief eigenplaces async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/eigenplaces-resnet50_512x512_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_eigenplaces_resnet50_512x512::EigenplacesFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::v_eigenplaces_resnet50_512x512::EigenplacesFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
