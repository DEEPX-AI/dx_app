/**
 * @file yolov8x_pose_sync.cpp
 * @brief Yolov8xPoseFactory synchronous inference example
 */

#include "factory/yolov8x_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8xPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolov8xPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
