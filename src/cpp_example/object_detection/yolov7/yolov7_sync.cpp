/**
 * @file yolov7_sync.cpp
 * @brief YOLOv7 synchronous inference example
 */

#include "factory/yolov7_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
