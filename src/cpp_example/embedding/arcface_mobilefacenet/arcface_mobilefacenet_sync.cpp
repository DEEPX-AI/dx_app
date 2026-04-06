/**
 * @file arcface_mobilefacenet_sync.cpp
 * @brief ArcFace MobileFaceNet synchronous embedding example
 */

#include "factory/arcface_mobilefacenet_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ArcFaceMobileFaceNetFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::ArcFaceMobileFaceNetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
