/**
 * @file deit_base_distilled_sync.cpp
 * @brief DeitBaseDistilled1Factory synchronous inference example
 */

#include "factory/deit_base_distilled_1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitBaseDistilled1Factory>();
    dxapp::SyncClassificationRunner<dxapp::DeitBaseDistilled1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
