/**
 * @file eigenplaces_resnet50_async.cpp
 * @brief EigenplacesResnet50Factory asynchronous inference example
 */

#include "factory/eigenplaces_resnet50_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EigenplacesResnet50Factory>();
    dxapp::AsyncEmbeddingRunner<dxapp::EigenplacesResnet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
