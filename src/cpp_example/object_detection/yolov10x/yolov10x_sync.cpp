/**
 * @file yolov10x_sync.cpp
 * @brief YOLOv10x synchronous inference example
 */

#include "factory/yolov10x_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10xFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv10xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
