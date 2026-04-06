/**
 * @file ssdmv2lite_async.cpp
 * @brief SSD MobileNet V2 Lite asynchronous inference example
 */

#include "factory/ssdmv2lite_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Ssdmv2liteFactory>();
    dxapp::AsyncDetectionRunner<dxapp::Ssdmv2liteFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
