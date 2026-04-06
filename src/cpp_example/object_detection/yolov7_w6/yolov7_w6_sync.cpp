/**
 * @file yolov7_w6_sync.cpp
 * @brief YOLOv7_w6 synchronous inference example
 */

#include "factory/yolov7_w6_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_w6Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7_w6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
