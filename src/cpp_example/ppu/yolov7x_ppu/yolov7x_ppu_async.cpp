/**
 * @file yolov7x_ppu_async.cpp
 * @brief YOLOv7X-PPU asynchronous inference example
 */

#include "factory/yolov7x_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7x_ppuFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7x_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
