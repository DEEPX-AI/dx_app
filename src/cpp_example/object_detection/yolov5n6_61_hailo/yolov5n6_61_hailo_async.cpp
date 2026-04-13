/**
 * @file yolov5n6_61_hailo_async.cpp
 * @brief YOLOv5n6_61_hailo asynchronous inference example
 */

#include "factory/yolov5n6_61_hailo_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5n6_61_hailoFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5n6_61_hailoFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
