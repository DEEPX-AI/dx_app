/**
 * @file yolov11n_ppu_sync.cpp
 * @brief YOLOv11N-PPU synchronous inference example
 */

#include "factory/yolov11n_ppu_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11n_ppuFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv11n_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
