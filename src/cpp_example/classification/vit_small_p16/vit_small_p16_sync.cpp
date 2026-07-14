/**
 * @file vit_small_p16_sync.cpp
 * @brief VitSmallP16Factory synchronous inference example
 */

#include "factory/vit_small_p16_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitSmallP16Factory>();
    dxapp::SyncClassificationRunner<dxapp::VitSmallP16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
