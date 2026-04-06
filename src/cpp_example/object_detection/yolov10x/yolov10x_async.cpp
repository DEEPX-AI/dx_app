/**
 * @file yolov10x_async.cpp
 * @brief YOLOv10x asynchronous inference example
 */

#include "factory/yolov10x_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10xFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv10xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
