/**
 * @file yolov7e6e_sync.cpp
 * @brief YOLOv7e6e synchronous inference example
 */

#include "factory/yolov7e6e_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7e6eFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7e6eFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
