/**
 * @file yolov10s_async.cpp
 * @brief YOLOv10s asynchronous inference example
 */

#include "factory/yolov10s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv10sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
