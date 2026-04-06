/**
 * @file yolov3_sync.cpp
 * @brief YOLOv3 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov3_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
