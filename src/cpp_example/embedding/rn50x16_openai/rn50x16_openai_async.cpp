/**
 * @file clip_rn50x16_async.cpp
 * @brief ClipRn50x16Factory asynchronous inference example
 */

#include "factory/rn50x16_openai_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ClipRn50x16Factory>();
    dxapp::AsyncEmbeddingRunner<dxapp::ClipRn50x16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
