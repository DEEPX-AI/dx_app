/**
 * @file clip_vit_l14_async.cpp
 * @brief ClipVitL14Factory asynchronous inference example
 */

#include "factory/vit_l_14_datacomp_xl_s13b_b90k_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ClipVitL14Factory>();
    dxapp::AsyncEmbeddingRunner<dxapp::ClipVitL14Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
