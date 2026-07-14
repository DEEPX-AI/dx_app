/**
 * @file depth_anything_v2_vits_sync.cpp
 * @brief DepthAnythingV2VitsFactory synchronous inference example
 */

#include "factory/depth_anything_v2_vits_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DepthAnythingV2VitsFactory>();
    dxapp::SyncDepthRunner<dxapp::DepthAnythingV2VitsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
