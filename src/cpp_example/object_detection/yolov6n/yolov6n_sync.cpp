/**
 * @file yolov6n_sync.cpp
 * @brief YOLOv6n synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov6n_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6nFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv6nFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
