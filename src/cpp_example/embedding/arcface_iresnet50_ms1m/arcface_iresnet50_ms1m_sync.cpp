/**
 * @file arcface_iresnet50_ms1m_sync.cpp
 * @brief ArcFace MobileFaceNet synchronous embedding example
 */

#include "factory/arcface_iresnet50_ms1m_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Arcface_iResNet50_ms1mFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::Arcface_iResNet50_ms1mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
