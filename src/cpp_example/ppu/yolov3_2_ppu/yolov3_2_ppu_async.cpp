/**
 * @file yolov3_2_ppu_async.cpp
 * @brief YOLOv3_2_ppu asynchronous inference example
 */

#include "factory/yolov3_2_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3_2_ppuFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv3_2_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
