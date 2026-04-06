/**
 * @file yolov5m_async.cpp
 * @brief YOLOv5m asynchronous inference example
 */

#include "factory/yolov5m_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5mFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
