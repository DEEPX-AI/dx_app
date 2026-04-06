/**
 * @file yolov9t_async.cpp
 * @brief YOLOv9t asynchronous inference example
 */

#include "factory/yolov9t_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9tFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv9tFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
