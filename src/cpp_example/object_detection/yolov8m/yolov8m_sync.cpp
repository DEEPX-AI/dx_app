/**
 * @file yolov8m_sync.cpp
 * @brief YOLOv8m synchronous inference example
 */

#include "factory/yolov8m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8mFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv8mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
