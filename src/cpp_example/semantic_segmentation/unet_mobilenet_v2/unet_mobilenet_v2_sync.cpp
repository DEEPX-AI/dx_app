/**
 * @file unet_mobilenet_v2_sync.cpp
 * @brief Unet_mobilenet_v2 synchronous semantic segmentation example
 *
 * Uses SyncSegmentationRunner for unified structure.
 */

#include "factory/unet_mobilenet_v2_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Unet_mobilenet_v2Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::Unet_mobilenet_v2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
