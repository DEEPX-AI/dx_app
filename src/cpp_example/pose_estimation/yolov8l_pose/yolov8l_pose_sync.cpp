/**
 * @file yolov8l_pose_sync.cpp
 * @brief Yolov8lPoseFactory synchronous inference example
 */

#include "factory/yolov8l_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8lPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolov8lPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
