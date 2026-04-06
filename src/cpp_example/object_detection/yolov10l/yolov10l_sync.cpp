/**
 * @file yolov10l_sync.cpp
 * @brief YOLOv10l synchronous inference example
 */

#include "factory/yolov10l_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10lFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv10lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
