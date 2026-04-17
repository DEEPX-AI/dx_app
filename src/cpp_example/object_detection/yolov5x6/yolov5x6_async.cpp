/**
 * @file yolov5x6_async.cpp
 * @brief YOLOv5x6 asynchronous inference example
 */

#include "factory/yolov5x6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5x6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5x6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
