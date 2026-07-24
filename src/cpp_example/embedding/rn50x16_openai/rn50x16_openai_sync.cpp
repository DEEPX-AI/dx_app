/**
 * @file clip_rn50x16_sync.cpp
 * @brief ClipRn50x16Factory synchronous inference example
 */

#include "factory/rn50x16_openai_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ClipRn50x16Factory>();
    dxapp::SyncEmbeddingRunner<dxapp::ClipRn50x16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
