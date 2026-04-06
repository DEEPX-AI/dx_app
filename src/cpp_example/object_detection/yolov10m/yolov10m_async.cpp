/**
 * @file yolov10m_async.cpp
 * @brief YOLOv10m asynchronous inference example
 */

#include "factory/yolov10m_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10mFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv10mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
