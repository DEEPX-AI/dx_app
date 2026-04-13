/**
 * @file yolov3_2_sync.cpp
 * @brief YOLOv3_2 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov3_2_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3_2Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv3_2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
