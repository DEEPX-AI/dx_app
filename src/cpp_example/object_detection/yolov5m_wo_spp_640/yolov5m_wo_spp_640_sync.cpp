/**
 * @file yolov5m_wo_spp_640_sync.cpp
 * @brief YOLOv5m_wo_spp_640 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5m_wo_spp_640_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_wo_spp_640Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5m_wo_spp_640Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
