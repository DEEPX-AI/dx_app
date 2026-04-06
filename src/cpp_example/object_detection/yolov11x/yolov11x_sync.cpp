/**
 * @file yolov11x_sync.cpp
 * @brief YOLOv11x synchronous inference example
 */

#include "factory/yolov11x_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11xFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv11xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
