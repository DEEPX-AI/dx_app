/**
 * @file yolov7e6_sync.cpp
 * @brief YOLOv7e6 synchronous inference example
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov7e6_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7e6Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7e6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
