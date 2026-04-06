/**
 * @file yolov8l_sync.cpp
 * @brief YOLOv8l synchronous inference example
 */

#include "factory/yolov8l_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8lFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv8lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
