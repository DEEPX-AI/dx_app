/**
 * @file yolov8s_async.cpp
 * @brief YOLOv8s asynchronous inference example
 */

#include "factory/yolov8s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv8sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
