/**
 * @file nanodetplusm_sync.cpp
 * @brief NanoDetplusm synchronous inference example
 */

#include "factory/nanodetplusm_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDetplusmFactory>();
    dxapp::SyncDetectionRunner<dxapp::NanoDetplusmFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
