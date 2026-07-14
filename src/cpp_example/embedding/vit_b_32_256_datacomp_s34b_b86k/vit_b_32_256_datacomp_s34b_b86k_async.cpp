/**
 * @file clip_vit_b32_256_async.cpp
 * @brief ClipVitB32256Factory asynchronous inference example
 */

#include "factory/vit_b_32_256_datacomp_s34b_b86k_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ClipVitB32256Factory>();
    dxapp::AsyncEmbeddingRunner<dxapp::ClipVitB32256Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
