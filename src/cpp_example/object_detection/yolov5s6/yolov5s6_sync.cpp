/**
 * @file yolov5s6_sync.cpp
 * @brief YOLOv5s6 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5s6_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s6Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5s6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
