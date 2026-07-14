/**
 * @file yolov8l_pose_async.cpp
 * @brief Yolov8lPoseFactory asynchronous inference example
 */

#include "factory/yolov8l_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8lPoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolov8lPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
