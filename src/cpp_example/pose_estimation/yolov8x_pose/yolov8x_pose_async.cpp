/**
 * @file yolov8x_pose_async.cpp
 * @brief Yolov8xPoseFactory asynchronous inference example
 */

#include "factory/yolov8x_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8xPoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolov8xPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
