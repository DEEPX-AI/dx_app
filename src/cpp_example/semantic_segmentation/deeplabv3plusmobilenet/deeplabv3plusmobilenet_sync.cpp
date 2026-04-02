/**
 * @file deeplabv3plusmobilenet_sync.cpp
 * @brief DeepLabv3 synchronous semantic segmentation example
 *
 * Part of DX-APP v3.0.0 refactoring.
 * Uses SyncSemanticSegRunner for unified structure.
 */

#include "factory/deeplabv3plusmobilenet_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeepLabv3Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::DeepLabv3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
