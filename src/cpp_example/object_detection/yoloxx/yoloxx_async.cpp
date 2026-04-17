/**
 * @file yoloxx_async.cpp
 * @brief YOLOXx asynchronous inference example
 */

#include "factory/yoloxx_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXxFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOXxFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
