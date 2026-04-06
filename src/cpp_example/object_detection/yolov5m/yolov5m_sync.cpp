/**
 * @file yolov5m_sync.cpp
 * @brief YOLOv5m synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5mFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
