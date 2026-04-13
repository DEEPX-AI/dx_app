/**
 * @file deeplabv3mobilenetv2_sync.cpp
 * @brief DeepLabv3mobilenetv2 synchronous semantic segmentation example
 *
 * Uses SyncSemanticSegRunner for unified structure.
 */

#include "factory/deeplabv3mobilenetv2_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeepLabv3mobilenetv2Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::DeepLabv3mobilenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
