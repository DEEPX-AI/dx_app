/**
 * @file yolov4_leaky_512_512_sync.cpp
 * @brief YOLOv4_leaky_512_512 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov4_leaky_512_512_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv4_leaky_512_512Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv4_leaky_512_512Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
