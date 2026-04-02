/**
 * @file yolov5s_sync.cpp
 * @brief YOLOv5s synchronous inference example
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5sFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
