/**
 * @file yolov5x_sync.cpp
 * @brief YOLOv5x synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5x_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5xFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
