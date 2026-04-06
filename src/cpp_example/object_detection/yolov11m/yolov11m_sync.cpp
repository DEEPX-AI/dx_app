/**
 * @file yolov11m_sync.cpp
 * @brief YOLOv11m synchronous inference example
 */

#include "factory/yolov11m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11mFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv11mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
