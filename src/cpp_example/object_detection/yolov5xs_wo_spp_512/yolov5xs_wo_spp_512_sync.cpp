/**
 * @file yolov5xs_wo_spp_512_sync.cpp
 * @brief YOLOv5xs_wo_spp_512 synchronous inference example
 * 
 * Uses abstract factory pattern for clean separation of concerns.
 */

#include "factory/yolov5xs_wo_spp_512_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5xs_wo_spp_512Factory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5xs_wo_spp_512Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
