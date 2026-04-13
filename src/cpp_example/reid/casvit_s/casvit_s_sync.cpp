/**
 * @file casvit_s_sync.cpp
 * @brief ArcFace MobileFaceNet synchronous embedding example
 */

#include "factory/casvit_s_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_sFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::Casvit_sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
