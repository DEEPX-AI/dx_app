/**
 * @file yolov9s_async.cpp
 * @brief YOLOv9s asynchronous inference example
 */

#include "factory/yolov9s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv9sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
