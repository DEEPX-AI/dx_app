/**
 * @file yolov5n6_async.cpp
 * @brief YOLOv5n6 asynchronous inference example
 */

#include "factory/yolov5n6_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5n6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5n6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
