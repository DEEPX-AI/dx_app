/**
 * @file centerpose_regnetx_800mf_sync.cpp
 * @brief Centerpose_regnetx_800mf synchronous inference example
 */

#include "factory/centerpose_regnetx_800mf_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Centerpose_regnetx_800mfFactory>();
    dxapp::SyncPoseRunner<dxapp::Centerpose_regnetx_800mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
