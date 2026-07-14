/**
 * @file deit_small_sync.cpp
 * @brief DeitSmallFactory synchronous inference example
 */

#include "factory/deit_small_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitSmallFactory>();
    dxapp::SyncClassificationRunner<dxapp::DeitSmallFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
