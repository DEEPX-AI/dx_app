/**
 * @file vit_pose_small_bn_async.cpp
 * @brief VitPoseSmallBnFactory asynchronous inference example
 */

#include "factory/vit_pose_small_bn_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitPoseSmallBnFactory>();
    dxapp::AsyncPoseRunner<dxapp::VitPoseSmallBnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
