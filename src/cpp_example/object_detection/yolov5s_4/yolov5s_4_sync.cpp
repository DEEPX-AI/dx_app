/**
 * @file yolov5s_4_sync.cpp
 * @brief YOLOv5s_4 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5s_4_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s_4Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5s_4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
