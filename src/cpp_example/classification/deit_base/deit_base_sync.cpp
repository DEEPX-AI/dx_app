/**
 * @file deit_base_sync.cpp
 * @brief DeitBaseFactory synchronous inference example
 */

#include "factory/deit_base_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitBaseFactory>();
    dxapp::SyncClassificationRunner<dxapp::DeitBaseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
