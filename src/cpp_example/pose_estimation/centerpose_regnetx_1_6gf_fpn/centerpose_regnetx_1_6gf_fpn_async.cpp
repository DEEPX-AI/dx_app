/**
 * @file centerpose_regnetx_1_6gf_fpn_async.cpp
 * @brief Centerpose_regnetx_1_6gf_fpn asynchronous inference example
 */

#include "factory/centerpose_regnetx_1_6gf_fpn_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Centerpose_regnetx_1_6gf_fpnFactory>();
    dxapp::AsyncPoseRunner<dxapp::Centerpose_regnetx_1_6gf_fpnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
