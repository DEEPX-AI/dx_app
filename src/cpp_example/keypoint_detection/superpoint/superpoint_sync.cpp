/**
 * @file superpoint_sync.cpp
 * @brief SuperPointFactory synchronous inference example
 */

#include "factory/superpoint_factory.hpp"
#include "common/runner/sync_keypoint_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SuperPointFactory>();
    dxapp::SyncKeypointRunner<dxapp::SuperPointFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
