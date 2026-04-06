/**
 * @file yolov8s_pose_async.cpp
 * @brief YOLOv8-Pose asynchronous inference example
 */

#include "factory/yolov8s_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8PoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::YOLOv8PoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
