/**
 * @file yolo11n_pose_sync.cpp
 * @brief Yolo11nPoseFactory synchronous inference example
 */

#include "factory/yolo11n_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11nPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolo11nPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
