/**
 * @file arcface_mobilefacenet_112x112
 * @brief arcface sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/arcface_mobilefacenet_112x112_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_arcface_mobilefacenet_112x112::ArcfaceFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::v_arcface_mobilefacenet_112x112::ArcfaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
