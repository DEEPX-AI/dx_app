/**
 * @file yolo26n_pose_sync.cpp
 * @brief Yolo26n_pose synchronous inference example
 */

#include "factory/yolo26n_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26n_poseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolo26n_poseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
