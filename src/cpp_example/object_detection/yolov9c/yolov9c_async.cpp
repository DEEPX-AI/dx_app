/**
 * @file yolov9c_async.cpp
 * @brief YOLOv9c asynchronous inference example
 */

#include "factory/yolov9c_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9cFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv9cFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
