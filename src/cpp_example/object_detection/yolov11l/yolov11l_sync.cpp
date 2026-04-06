/**
 * @file yolov11l_sync.cpp
 * @brief YOLOv11l synchronous inference example
 */

#include "factory/yolov11l_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11lFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv11lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
