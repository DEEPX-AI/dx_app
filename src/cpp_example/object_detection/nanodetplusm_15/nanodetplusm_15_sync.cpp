/**
 * @file nanodetplusm_15_sync.cpp
 * @brief NanoDetplusm_15 synchronous inference example
 */

#include "factory/nanodetplusm_15_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDetplusm_15Factory>();
    dxapp::SyncDetectionRunner<dxapp::NanoDetplusm_15Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
