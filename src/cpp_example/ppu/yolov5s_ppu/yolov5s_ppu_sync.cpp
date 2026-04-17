/**
 * @file yolov5s_ppu_sync.cpp
 * @brief YOLOv5-PPU synchronous inference example
 */

#include "factory/yolov5s_ppu_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s_ppuFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv5s_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
