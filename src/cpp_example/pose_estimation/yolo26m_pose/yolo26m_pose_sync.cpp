/**
 * @file yolo26m_pose_sync.cpp
 * @brief Yolo26m_pose synchronous inference example
 */

#include "factory/yolo26m_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26m_poseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolo26m_poseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
