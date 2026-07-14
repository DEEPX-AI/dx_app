/**
 * @file vit_base_p16_sync.cpp
 * @brief VitBaseP16Factory synchronous inference example
 */

#include "factory/vit_base_p16_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitBaseP16Factory>();
    dxapp::SyncClassificationRunner<dxapp::VitBaseP16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
