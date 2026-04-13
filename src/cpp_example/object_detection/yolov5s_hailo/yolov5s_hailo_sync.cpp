/**
 * @file yolov5s_hailo_sync.cpp
 * @brief YOLOv5s_hailo synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5s_hailo_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s_hailoFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5s_hailoFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
