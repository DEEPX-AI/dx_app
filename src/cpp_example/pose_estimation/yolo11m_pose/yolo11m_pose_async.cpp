/**
 * @file yolo11m_pose_async.cpp
 * @brief Yolo11mPoseFactory asynchronous inference example
 */

#include "factory/yolo11m_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11mPoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolo11mPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
