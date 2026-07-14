/**
 * @file yolov6s_async.cpp
 * @brief YOLOv6s asynchronous inference example
 */

#include "factory/yolov6s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv6sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
