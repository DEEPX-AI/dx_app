/**
 * @file yolov11m_async.cpp
 * @brief YOLOv11m asynchronous inference example
 */

#include "factory/yolov11m_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11mFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv11mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
