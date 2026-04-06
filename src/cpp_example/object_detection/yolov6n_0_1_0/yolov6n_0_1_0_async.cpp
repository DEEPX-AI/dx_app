/**
 * @file yolov6n_0_1_0_async.cpp
 * @brief YOLOv6n_0_1_0 asynchronous inference example
 */

#include "factory/yolov6n_0_1_0_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6n_0_1_0Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv6n_0_1_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
