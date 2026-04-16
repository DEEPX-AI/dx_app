/**
 * @file yolov5m6_61_h_sync.cpp
 * @brief YOLOv5m6_61_h synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5m6_61_h_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m6_61_hFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5m6_61_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
