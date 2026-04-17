/**
 * @file yolov7_ppu_async.cpp
 * @brief Asynchronous inference example for YOLOv7-PPU model
 */

#include "factory/yolov7_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7PPUFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7PPUFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
