/**
 * @file yolov9c_sync.cpp
 * @brief YOLOv9c synchronous inference example
 */

#include "factory/yolov9c_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9cFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv9cFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
