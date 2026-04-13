/**
 * @file yolov6m_sync.cpp
 * @brief YOLOv6m synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov6m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6mFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv6mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
