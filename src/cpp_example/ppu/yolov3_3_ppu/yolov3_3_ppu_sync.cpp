/**
 * @file yolov3_3_ppu_sync.cpp
 * @brief YOLOv3_3_ppu synchronous inference example
 */

#include "factory/yolov3_3_ppu_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3_3_ppuFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv3_3_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
