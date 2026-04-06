/**
 * @file yolov6n_0_2_1_async.cpp
 * @brief YOLOv6n_0_2_1 asynchronous inference example
 */

#include "factory/yolov6n_0_2_1_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6n_0_2_1Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv6n_0_2_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
