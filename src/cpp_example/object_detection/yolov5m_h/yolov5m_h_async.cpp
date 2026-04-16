/**
 * @file yolov5m_h_async.cpp
 * @brief YOLOv5m_h asynchronous inference example
 */

#include "factory/yolov5m_h_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_hFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5m_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
