/**
 * @file deeplabv3plus_drn_512x512_sync.cpp
 * @brief DeepLabV3+ DRN 512x512 synchronous inference example
 */

#include "factory/deeplabv3plus_drn_512x512_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deeplabv3plusDrn512x512Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::Deeplabv3plusDrn512x512Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
