/**
 * @file beit_large_p16_sync.cpp
 * @brief BeitLargeP16Factory synchronous inference example
 */

#include "factory/beit_large_patch16_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::BeitLargeP16Factory>();
    dxapp::SyncClassificationRunner<dxapp::BeitLargeP16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
