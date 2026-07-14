/**
 * @file yolo11n_pose_async.cpp
 * @brief Yolo11nPoseFactory asynchronous inference example
 */

#include "factory/yolo11n_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11nPoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolo11nPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
