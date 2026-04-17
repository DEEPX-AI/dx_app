/**
 * @file yolov5s6_61_h_async.cpp
 * @brief YOLOv5s6_61_h asynchronous inference example
 */

#include "factory/yolov5s6_61_h_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s6_61_hFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5s6_61_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
