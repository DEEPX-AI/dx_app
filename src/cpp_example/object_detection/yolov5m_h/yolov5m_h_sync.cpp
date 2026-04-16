/**
 * @file yolov5m_h_sync.cpp
 * @brief YOLOv5m_h synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5m_h_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_hFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5m_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
