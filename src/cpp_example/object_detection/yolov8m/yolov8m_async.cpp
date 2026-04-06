/**
 * @file yolov8m_async.cpp
 * @brief YOLOv8m asynchronous inference example
 */

#include "factory/yolov8m_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8mFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv8mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
