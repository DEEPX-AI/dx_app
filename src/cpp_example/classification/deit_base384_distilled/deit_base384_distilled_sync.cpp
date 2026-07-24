/**
 * @file deit_base384_distilled_sync.cpp
 * @brief DeitBase384DistilledFactory synchronous inference example
 */

#include "factory/deit_base384_distilled_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitBase384DistilledFactory>();
    dxapp::SyncClassificationRunner<dxapp::DeitBase384DistilledFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
