/**
 * @file yolov8n_ppu_async.cpp
 * @brief YOLOv8N-PPU asynchronous inference example
 */

#include "factory/yolov8n_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8n_ppuFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv8n_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
