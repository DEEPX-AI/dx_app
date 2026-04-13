/**
 * @file yolov7d6_async.cpp
 * @brief YOLOv7d6 asynchronous inference example
 */

#include "factory/yolov7d6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7d6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7d6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
