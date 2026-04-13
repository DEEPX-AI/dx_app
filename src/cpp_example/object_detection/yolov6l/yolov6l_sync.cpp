/**
 * @file yolov6l_sync.cpp
 * @brief YOLOv6l synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov6l_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6lFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv6lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
