/**
 * @file yolo26m_pose_async.cpp
 * @brief Yolo26m_pose asynchronous inference example
 */

#include "factory/yolo26m_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26m_poseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolo26m_poseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
