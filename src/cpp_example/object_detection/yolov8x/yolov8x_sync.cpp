/**
 * @file yolov8x_sync.cpp
 * @brief YOLOv8x synchronous inference example
 */

#include "factory/yolov8x_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8xFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv8xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
