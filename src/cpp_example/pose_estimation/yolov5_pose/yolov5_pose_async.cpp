/**
 * @file yolov5_pose_async.cpp
 * @brief Yolov5PoseFactory asynchronous inference example
 */

#include "factory/yolov5_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov5PoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolov5PoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
