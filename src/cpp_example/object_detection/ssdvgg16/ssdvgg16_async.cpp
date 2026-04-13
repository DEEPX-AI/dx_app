/**
 * @file ssdvgg16_async.cpp
 * @brief SSD MobileNet V1 asynchronous inference example
 */

#include "factory/ssdvgg16_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SSDvgg16Factory>();
    dxapp::AsyncDetectionRunner<dxapp::SSDvgg16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
