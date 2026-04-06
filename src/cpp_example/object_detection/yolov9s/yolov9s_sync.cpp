/**
 * @file yolov9s_sync.cpp
 * @brief YOLOv9s synchronous inference example
 */

#include "factory/yolov9s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9sFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv9sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
