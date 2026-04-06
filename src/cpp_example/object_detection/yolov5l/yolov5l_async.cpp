/**
 * @file yolov5l_async.cpp
 * @brief YOLOv5l asynchronous inference example
 */

#include "factory/yolov5l_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5lFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
