/**
 * @file yolov3_2_async.cpp
 * @brief YOLOv3_2 asynchronous inference example
 */

#include "factory/yolov3_2_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3_2Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv3_2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
