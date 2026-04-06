/**
 * @file yolov11s_sync.cpp
 * @brief YOLOv11s synchronous inference example
 */

#include "factory/yolov11s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11sFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv11sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
