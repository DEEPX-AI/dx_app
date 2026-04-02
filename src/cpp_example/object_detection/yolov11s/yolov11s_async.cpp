/**
 * @file yolov11s_async.cpp
 * @brief YOLOv11s asynchronous inference example
 */

#include "factory/yolov11s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv11sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
