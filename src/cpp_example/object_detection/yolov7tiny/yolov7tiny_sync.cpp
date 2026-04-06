/**
 * @file yolov7tiny_sync.cpp
 * @brief YOLOv7tiny synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov7tiny_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7tinyFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7tinyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
