/**
 * @file yolov11n_sync.cpp
 * @brief YOLOv11 synchronous inference example
 */

#include "factory/yolov11n_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv11Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
