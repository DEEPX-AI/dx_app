/**
 * @file yolov8n_sync.cpp
 * @brief YOLOv8 synchronous inference example
 */

#include "factory/yolov8n_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv8Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
