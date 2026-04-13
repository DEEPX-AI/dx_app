/**
 * @file yolov3tiny_1_sync.cpp
 * @brief YOLOv3tiny_1 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov3tiny_1_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3tiny_1Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv3tiny_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
