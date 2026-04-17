/**
 * @file yolov8s_ppu_sync.cpp
 * @brief YOLOv8S-PPU synchronous inference example
 */

#include "factory/yolov8s_ppu_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8s_ppuFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv8s_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
