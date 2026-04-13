/**
 * @file arcface_r50_sync.cpp
 * @brief ArcFace MobileFaceNet synchronous embedding example
 */

#include "factory/arcface_r50_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Arcface_r50Factory>();
    dxapp::SyncEmbeddingRunner<dxapp::Arcface_r50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
