/**
 * @file yolov9t_ppu_async.cpp
 * @brief YOLOv9T-PPU asynchronous inference example
 */

#include "factory/yolov9t_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9t_ppuFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv9t_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
