/**
 * @file yolov8m_pose_sync.cpp
 * @brief YOLOv8-Pose synchronous inference example
 */

#include "factory/yolov8m_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8m_poseFactory>();
    dxapp::SyncPoseRunner<dxapp::YOLOv8m_poseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
