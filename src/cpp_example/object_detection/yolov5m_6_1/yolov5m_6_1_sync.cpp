/**
 * @file yolov5m_6_1_sync.cpp
 * @brief YOLOv5m_6_1 synchronous inference example
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5m_6_1_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_6_1Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5m_6_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
