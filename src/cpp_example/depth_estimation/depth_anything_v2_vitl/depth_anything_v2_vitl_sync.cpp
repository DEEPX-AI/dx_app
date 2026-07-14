/**
 * @file depth_anything_v2_vitl_sync.cpp
 * @brief DepthAnythingV2VitlFactory synchronous inference example
 */

#include "factory/depth_anything_v2_vitl_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DepthAnythingV2VitlFactory>();
    dxapp::SyncDepthRunner<dxapp::DepthAnythingV2VitlFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
