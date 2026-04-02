/**
 * @file yolov8m_pose_async.cpp
 * @brief YOLOv8-Pose asynchronous inference example
 */

#include "factory/yolov8m_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8m_poseFactory>();
    dxapp::AsyncPoseRunner<dxapp::YOLOv8m_poseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
