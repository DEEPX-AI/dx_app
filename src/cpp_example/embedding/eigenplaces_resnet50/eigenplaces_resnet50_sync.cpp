/**
 * @file eigenplaces_resnet50_sync.cpp
 * @brief EigenplacesResnet50Factory synchronous inference example
 */

#include "factory/eigenplaces_resnet50_factory.hpp"
#include "common/runner/sync_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EigenplacesResnet50Factory>();
    dxapp::SyncEmbeddingRunner<dxapp::EigenplacesResnet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
