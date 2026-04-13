/**
 * @file casvit_xs_sync.cpp
 * @brief ArcFace MobileFaceNet synchronous embedding example
 */

#include "factory/casvit_xs_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_xsFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::Casvit_xsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
