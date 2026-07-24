/**
 * @file deit_base_distilled_sync.cpp
 * @brief DeitBaseDistilled2Factory synchronous inference example
 */

#include "factory/deit_base_distilled_2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitBaseDistilled2Factory>();
    dxapp::SyncClassificationRunner<dxapp::DeitBaseDistilled2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
