/**
 * @file ghostnet_sync.cpp
 * @brief GhostnetFactory synchronous inference example
 */

#include "factory/ghostnet_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::GhostnetFactory>();
    dxapp::SyncClassificationRunner<dxapp::GhostnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
