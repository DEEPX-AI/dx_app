/**
 * @file yolo11s_pose_sync.cpp
 * @brief Yolo11sPoseFactory synchronous inference example
 */

#include "factory/yolo11s_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11sPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolo11sPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
