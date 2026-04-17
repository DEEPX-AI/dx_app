/**
 * @file yolov8x_async.cpp
 * @brief YOLOv8x asynchronous inference example
 */

#include "factory/yolov8x_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8xFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv8xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
