/**
 * @file casvit_t_sync.cpp
 * @brief ArcFace MobileFaceNet synchronous embedding example
 */

#include "factory/casvit_t_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_tFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::Casvit_tFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
