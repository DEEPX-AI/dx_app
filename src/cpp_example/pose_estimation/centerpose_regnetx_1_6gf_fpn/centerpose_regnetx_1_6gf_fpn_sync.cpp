/**
 * @file centerpose_regnetx_1_6gf_fpn_sync.cpp
 * @brief Centerpose_regnetx_1_6gf_fpn synchronous inference example
 */

#include "factory/centerpose_regnetx_1_6gf_fpn_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Centerpose_regnetx_1_6gf_fpnFactory>();
    dxapp::SyncPoseRunner<dxapp::Centerpose_regnetx_1_6gf_fpnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
