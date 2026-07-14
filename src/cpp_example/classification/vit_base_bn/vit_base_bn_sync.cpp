/**
 * @file vit_base_bn_sync.cpp
 * @brief VitBaseBnFactory synchronous inference example
 */

#include "factory/vit_base_bn_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitBaseBnFactory>();
    dxapp::SyncClassificationRunner<dxapp::VitBaseBnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
