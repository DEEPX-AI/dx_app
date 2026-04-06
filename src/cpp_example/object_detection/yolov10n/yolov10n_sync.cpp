/**
 * @file yolov10n_sync.cpp
 * @brief YOLOv10 synchronous inference example
 */

#include "factory/yolov10n_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv10Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
