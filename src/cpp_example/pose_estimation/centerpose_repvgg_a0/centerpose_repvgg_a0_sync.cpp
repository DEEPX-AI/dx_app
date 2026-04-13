/**
 * @file centerpose_repvgg_a0_sync.cpp
 * @brief Centerpose_repvgg_a0 synchronous inference example
 */

#include "factory/centerpose_repvgg_a0_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Centerpose_repvgg_a0Factory>();
    dxapp::SyncPoseRunner<dxapp::Centerpose_repvgg_a0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
