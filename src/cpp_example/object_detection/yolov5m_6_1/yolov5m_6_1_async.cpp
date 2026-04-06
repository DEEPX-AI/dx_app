/**
 * @file yolov5m_6_1_async.cpp
 * @brief YOLOv5m_6_1 asynchronous inference example
 */

#include "factory/yolov5m_6_1_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_6_1Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5m_6_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
