/**
 * @file yolov5l_h_sync.cpp
 * @brief YOLOv5l_h synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5l_h_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5l_hFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5l_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
