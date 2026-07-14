/**
 * @file deit_small_distilled_sync.cpp
 * @brief DeitSmallDistilledFactory synchronous inference example
 */

#include "factory/deit_small_distilled_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitSmallDistilledFactory>();
    dxapp::SyncClassificationRunner<dxapp::DeitSmallDistilledFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
