/**
 * @file yolo11x_pose_async.cpp
 * @brief Yolo11xPoseFactory asynchronous inference example
 */

#include "factory/yolo11x_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11xPoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolo11xPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
