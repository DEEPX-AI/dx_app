/**
 * @file yolov5l6_async.cpp
 * @brief YOLOv5l6 asynchronous inference example
 */

#include "factory/yolov5l6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5l6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5l6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
