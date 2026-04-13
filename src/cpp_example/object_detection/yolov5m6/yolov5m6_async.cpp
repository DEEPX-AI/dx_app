/**
 * @file yolov5m6_async.cpp
 * @brief YOLOv5m6 asynchronous inference example
 */

#include "factory/yolov5m6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5m6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
