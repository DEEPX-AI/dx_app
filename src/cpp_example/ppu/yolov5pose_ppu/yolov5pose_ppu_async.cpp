/**
 * @file yolov5pose_ppu_async.cpp
 * @brief YOLOv5Pose-PPU asynchronous inference example
 */

#include "factory/yolov5pose_ppu_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5pose_ppuFactory>();
    dxapp::AsyncPoseRunner<dxapp::YOLOv5pose_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
