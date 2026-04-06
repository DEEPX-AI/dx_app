/**
 * @file yolov7_async.cpp
 * @brief YOLOv7 asynchronous inference example
 */

#include "factory/yolov7_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
