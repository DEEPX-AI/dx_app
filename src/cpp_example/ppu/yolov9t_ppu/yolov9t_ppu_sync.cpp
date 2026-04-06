/**
 * @file yolov9t_ppu_sync.cpp
 * @brief YOLOv9T-PPU synchronous inference example
 */

#include "factory/yolov9t_ppu_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9t_ppuFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv9t_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
