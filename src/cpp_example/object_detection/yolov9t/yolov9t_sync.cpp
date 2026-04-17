/**
 * @file yolov9t_sync.cpp
 * @brief YOLOv9t synchronous inference example
 */

#include "factory/yolov9t_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9tFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv9tFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
