/**
 * @file yolov7x_sync.cpp
 * @brief YOLOv7x synchronous inference example
 */

#include "factory/yolov7x_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7xFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
