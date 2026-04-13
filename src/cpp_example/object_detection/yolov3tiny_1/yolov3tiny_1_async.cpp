/**
 * @file yolov3tiny_1_async.cpp
 * @brief YOLOv3tiny_1 asynchronous inference example
 */

#include "factory/yolov3tiny_1_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3tiny_1Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv3tiny_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
