/**
 * @file casvit_t_seg_fpn_resnet50_sync.cpp
 * @brief CasvitTSegFpnResnet50Factory synchronous inference example
 */

#include "factory/casvit_t_seg_fpn_resnet50_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::CasvitTSegFpnResnet50Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::CasvitTSegFpnResnet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
