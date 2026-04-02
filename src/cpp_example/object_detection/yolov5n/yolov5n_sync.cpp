/**
 * @file yolov5n_sync.cpp
 * @brief YOLOv5 synchronous inference example
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5n_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
