/**
 * @file yolov5s6_async.cpp
 * @brief YOLOv5s6 asynchronous inference example
 */

#include "factory/yolov5s6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5s6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
