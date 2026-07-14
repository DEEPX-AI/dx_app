/**
 * @file vit_pose_small_bn_sync.cpp
 * @brief VitPoseSmallBnFactory synchronous inference example
 */

#include "factory/vit_pose_small_bn_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitPoseSmallBnFactory>();
    dxapp::SyncPoseRunner<dxapp::VitPoseSmallBnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
