/**
 * @file yolov6n_0_2_1_nms_core_sync.cpp
 * @brief YOLOv6n_0_2_1_nms_core synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov6n_0_2_1_nms_core_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6n_0_2_1_nms_coreFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv6n_0_2_1_nms_coreFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
