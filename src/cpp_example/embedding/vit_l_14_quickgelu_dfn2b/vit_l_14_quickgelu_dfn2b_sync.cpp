/**
 * @file clip_vit_l14_quickgelu_sync.cpp
 * @brief ClipVitL14QuickgeluFactory synchronous inference example
 */

#include "factory/vit_l_14_quickgelu_dfn2b_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ClipVitL14QuickgeluFactory>();
    dxapp::SyncEmbeddingRunner<dxapp::ClipVitL14QuickgeluFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
