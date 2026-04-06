/**
 * @file yolov10m_sync.cpp
 * @brief YOLOv10m synchronous inference example
 */

#include "factory/yolov10m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10mFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv10mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
