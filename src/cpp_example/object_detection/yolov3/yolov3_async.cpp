/**
 * @file yolov3_async.cpp
 * @brief YOLOv3 asynchronous inference example
 */

#include "factory/yolov3_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
