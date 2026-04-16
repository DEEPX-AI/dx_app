/**
 * @file yolov5s_h_async.cpp
 * @brief YOLOv5s_h asynchronous inference example
 */

#include "factory/yolov5s_h_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s_hFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5s_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
