/**
 * @file yolov11n_async.cpp
 * @brief YOLOv11 asynchronous inference example
 */

#include "factory/yolov11n_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv11Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
