/**
 * @file yolov6n_async.cpp
 * @brief YOLOv6n asynchronous inference example
 */

#include "factory/yolov6n_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6nFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv6nFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
