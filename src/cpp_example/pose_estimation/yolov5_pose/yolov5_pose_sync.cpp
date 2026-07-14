/**
 * @file yolov5_pose_sync.cpp
 * @brief Yolov5PoseFactory synchronous inference example
 */

#include "factory/yolov5_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov5PoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolov5PoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
