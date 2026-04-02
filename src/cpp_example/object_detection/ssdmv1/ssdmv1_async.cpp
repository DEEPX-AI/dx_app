/**
 * @file ssdmv1_async.cpp
 * @brief SSD MobileNet V1 asynchronous inference example
 */

#include "factory/ssdmv1_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SSDmv1Factory>();
    dxapp::AsyncDetectionRunner<dxapp::SSDmv1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
