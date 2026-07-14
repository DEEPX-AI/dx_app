/**
 * @file yolo11m_pose_sync.cpp
 * @brief Yolo11mPoseFactory synchronous inference example
 */

#include "factory/yolo11m_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11mPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolo11mPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
