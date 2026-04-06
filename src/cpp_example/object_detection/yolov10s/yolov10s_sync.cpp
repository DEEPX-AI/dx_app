/**
 * @file yolov10s_sync.cpp
 * @brief YOLOv10s synchronous inference example
 */

#include "factory/yolov10s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10sFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv10sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
