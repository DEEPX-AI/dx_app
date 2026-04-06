/**
 * @file yolov5s_ppu_async.cpp
 * @brief YOLOv5-PPU asynchronous inference example
 */

#include "factory/yolov5s_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s_ppuFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5s_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
