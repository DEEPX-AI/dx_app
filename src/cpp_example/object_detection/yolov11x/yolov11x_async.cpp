/**
 * @file yolov11x_async.cpp
 * @brief YOLOv11x asynchronous inference example
 */

#include "factory/yolov11x_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11xFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv11xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
