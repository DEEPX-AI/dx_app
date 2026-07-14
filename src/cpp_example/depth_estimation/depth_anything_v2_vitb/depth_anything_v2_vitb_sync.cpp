/**
 * @file depth_anything_v2_vitb_sync.cpp
 * @brief DepthAnythingV2VitbFactory synchronous inference example
 */

#include "factory/depth_anything_v2_vitb_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DepthAnythingV2VitbFactory>();
    dxapp::SyncDepthRunner<dxapp::DepthAnythingV2VitbFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
