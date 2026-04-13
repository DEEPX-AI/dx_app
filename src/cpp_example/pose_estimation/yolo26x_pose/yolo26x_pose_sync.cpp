/**
 * @file yolo26x_pose_sync.cpp
 * @brief Yolo26x_pose synchronous inference example
 */

#include "factory/yolo26x_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26x_poseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolo26x_poseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
