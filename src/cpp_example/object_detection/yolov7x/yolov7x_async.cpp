/**
 * @file yolov7x_async.cpp
 * @brief YOLOv7x asynchronous inference example
 */

#include "factory/yolov7x_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7xFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
