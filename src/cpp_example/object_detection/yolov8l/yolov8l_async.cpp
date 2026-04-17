/**
 * @file yolov8l_async.cpp
 * @brief YOLOv8l asynchronous inference example
 */

#include "factory/yolov8l_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8lFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv8lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
