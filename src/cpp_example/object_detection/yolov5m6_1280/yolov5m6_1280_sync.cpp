/**
 * @file yolov5m6_1280_sync.cpp
 * @brief YOLOv5m6_1280 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5m6_1280_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m6_1280Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5m6_1280Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
