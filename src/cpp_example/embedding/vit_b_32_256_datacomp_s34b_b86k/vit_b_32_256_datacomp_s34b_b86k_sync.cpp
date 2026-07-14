/**
 * @file clip_vit_b32_256_sync.cpp
 * @brief ClipVitB32256Factory synchronous inference example
 */

#include "factory/vit_b_32_256_datacomp_s34b_b86k_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ClipVitB32256Factory>();
    dxapp::SyncEmbeddingRunner<dxapp::ClipVitB32256Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
