/**
 * @file yolov8n_pose_sync.cpp
 * @brief Yolov8nPoseFactory synchronous inference example
 */

#include "factory/yolov8n_pose_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8nPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::Yolov8nPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
