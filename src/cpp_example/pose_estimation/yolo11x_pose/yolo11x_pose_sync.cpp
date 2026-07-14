/**
 * @file yolo11x_pose_sync.cpp
 * @brief Yolo11xPoseFactory synchronous inference example
 */

#include "factory/yolo11x_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11xPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolo11xPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
