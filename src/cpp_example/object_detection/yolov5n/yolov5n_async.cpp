/**
 * @file yolov5n_async.cpp
 * @brief YOLOv5 asynchronous inference example
 */

#include "factory/yolov5n_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
