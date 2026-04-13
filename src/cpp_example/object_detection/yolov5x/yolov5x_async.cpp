/**
 * @file yolov5x_async.cpp
 * @brief YOLOv5x asynchronous inference example
 */

#include "factory/yolov5x_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5xFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
