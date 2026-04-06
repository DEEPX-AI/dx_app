/**
 * @file yolov10b_sync.cpp
 * @brief YOLOv10b synchronous inference example
 */

#include "factory/yolov10b_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10bFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv10bFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
