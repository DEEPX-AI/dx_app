/**
 * @file eigenplaces_resnet18_sync.cpp
 * @brief EigenplacesResnet18Factory synchronous inference example
 */

#include "factory/eigenplaces_resnet18_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EigenplacesResnet18Factory>();
    dxapp::SyncEmbeddingRunner<dxapp::EigenplacesResnet18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
