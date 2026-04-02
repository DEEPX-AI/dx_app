/**
 * @file yolov5l_sync.cpp
 * @brief YOLOv5l synchronous inference example
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5l_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5lFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
