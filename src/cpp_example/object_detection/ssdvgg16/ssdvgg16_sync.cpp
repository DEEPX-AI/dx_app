/**
 * @file ssdvgg16_sync.cpp
 * @brief SSD MobileNet V1 synchronous inference example
 */

#include "factory/ssdvgg16_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SSDvgg16Factory>();
    dxapp::SyncDetectionRunner<dxapp::SSDvgg16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
