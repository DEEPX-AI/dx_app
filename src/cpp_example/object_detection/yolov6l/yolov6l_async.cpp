/**
 * @file yolov6l_async.cpp
 * @brief YOLOv6l asynchronous inference example
 */

#include "factory/yolov6l_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6lFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv6lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
