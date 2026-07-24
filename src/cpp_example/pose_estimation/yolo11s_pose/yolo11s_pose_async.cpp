/**
 * @file yolo11s_pose_async.cpp
 * @brief Yolo11sPoseFactory asynchronous inference example
 */

#include "factory/yolo11s_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11sPoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolo11sPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
