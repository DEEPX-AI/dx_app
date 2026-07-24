/**
 * @file superpoint_async.cpp
 * @brief SuperPointFactory asynchronous inference example
 */

#include "factory/superpoint_factory.hpp"
#include "common/runner/async_keypoint_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SuperPointFactory>();
    dxapp::AsyncKeypointRunner<dxapp::SuperPointFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
