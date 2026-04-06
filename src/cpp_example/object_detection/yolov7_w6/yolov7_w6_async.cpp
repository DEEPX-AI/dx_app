/**
 * @file yolov7_w6_async.cpp
 * @brief YOLOv7_w6 asynchronous inference example
 */

#include "factory/yolov7_w6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_w6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7_w6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
