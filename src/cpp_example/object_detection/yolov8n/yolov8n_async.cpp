/**
 * @file yolov8n_async.cpp
 * @brief YOLOv8 asynchronous inference example
 */

#include "factory/yolov8n_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv8Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
