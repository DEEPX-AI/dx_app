/**
 * @file casvit_m_sync.cpp
 * @brief ArcFace MobileFaceNet synchronous embedding example
 */

#include "factory/casvit_m_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_mFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::Casvit_mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
