/**
 * @file yolov5pose_ppu_sync.cpp
 * @brief YOLOv5Pose-PPU synchronous inference example
 */

#include "factory/yolov5pose_ppu_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5pose_ppuFactory>();
    dxapp::SyncPoseRunner<dxapp::YOLOv5pose_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
