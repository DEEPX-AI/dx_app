/**
 * @file eigenplaces_resnet18_async.cpp
 * @brief EigenplacesResnet18Factory asynchronous inference example
 */

#include "factory/eigenplaces_resnet18_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EigenplacesResnet18Factory>();
    dxapp::AsyncEmbeddingRunner<dxapp::EigenplacesResnet18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
