/**
 * @file deeplabv3plusmobilenetv2_sync.cpp
 * @brief DeepLabv3plusmobilenetv2 synchronous semantic segmentation example
 *
 * Uses SyncSemanticSegRunner for unified structure.
 */

#include "factory/deeplabv3plusmobilenetv2_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeepLabv3plusmobilenetv2Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::DeepLabv3plusmobilenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
