/**
 * @file yolov7e6_async.cpp
 * @brief YOLOv7e6 asynchronous inference example
 */

#include "factory/yolov7e6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7e6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7e6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
