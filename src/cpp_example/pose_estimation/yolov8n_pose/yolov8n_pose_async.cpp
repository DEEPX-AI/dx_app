/**
 * @file yolov8n_pose_async.cpp
 * @brief Yolov8nPoseFactory asynchronous inference example
 */

#include "factory/yolov8n_pose_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8nPoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::Yolov8nPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
