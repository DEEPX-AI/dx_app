/**
 * @file yolov7x_ppu_sync.cpp
 * @brief YOLOv7X-PPU synchronous inference example
 */

#include "factory/yolov7x_ppu_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7x_ppuFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7x_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
