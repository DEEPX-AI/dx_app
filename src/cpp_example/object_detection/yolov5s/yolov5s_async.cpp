/**
 * @file yolov5s_async.cpp
 * @brief YOLOv5s asynchronous inference example
 */

#include "factory/yolov5s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
