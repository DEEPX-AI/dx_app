/**
 * @file yolov6s_sync.cpp
 * @brief YOLOv6s synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov6s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6sFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv6sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
