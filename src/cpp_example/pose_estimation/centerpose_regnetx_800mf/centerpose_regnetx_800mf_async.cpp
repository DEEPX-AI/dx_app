/**
 * @file centerpose_regnetx_800mf_async.cpp
 * @brief Centerpose_regnetx_800mf asynchronous inference example
 */

#include "factory/centerpose_regnetx_800mf_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Centerpose_regnetx_800mfFactory>();
    dxapp::AsyncPoseRunner<dxapp::Centerpose_regnetx_800mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
