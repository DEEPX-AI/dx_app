/**
 * @file ssdmv2lite_sync.cpp
 * @brief SSD MobileNet V2 Lite synchronous inference example
 */

#include "factory/ssdmv2lite_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Ssdmv2liteFactory>();
    dxapp::SyncDetectionRunner<dxapp::Ssdmv2liteFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
