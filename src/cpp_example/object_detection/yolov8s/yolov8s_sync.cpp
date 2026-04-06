/**
 * @file yolov8s_sync.cpp
 * @brief YOLOv8s synchronous inference example
 */

#include "factory/yolov8s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8sFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv8sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
