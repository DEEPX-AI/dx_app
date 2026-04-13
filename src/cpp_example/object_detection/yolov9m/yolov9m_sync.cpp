/**
 * @file yolov9m_sync.cpp
 * @brief YOLOv9m synchronous inference example
 */

#include "factory/yolov9m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9mFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv9mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
