/**
 * @file yoloxtiny_async.cpp
 * @brief YOLOXtiny asynchronous inference example
 */

#include "factory/yoloxtiny_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXtinyFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOXtinyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
