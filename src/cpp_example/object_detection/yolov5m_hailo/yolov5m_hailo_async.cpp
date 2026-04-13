/**
 * @file yolov5m_hailo_async.cpp
 * @brief YOLOv5m_hailo asynchronous inference example
 */

#include "factory/yolov5m_hailo_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_hailoFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5m_hailoFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
