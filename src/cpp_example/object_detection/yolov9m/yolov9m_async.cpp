/**
 * @file yolov9m_async.cpp
 * @brief YOLOv9m asynchronous inference example
 */

#include "factory/yolov9m_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9mFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv9mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
