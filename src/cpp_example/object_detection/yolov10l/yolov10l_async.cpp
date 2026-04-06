/**
 * @file yolov10l_async.cpp
 * @brief YOLOv10l asynchronous inference example
 */

#include "factory/yolov10l_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10lFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv10lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
