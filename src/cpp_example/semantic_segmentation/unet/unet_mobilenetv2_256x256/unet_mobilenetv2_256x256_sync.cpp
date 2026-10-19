/**
 * @file unet_mobilenetv2_256x256
 * @brief unet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/unet_mobilenetv2_256x256_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_unet_mobilenetv2_256x256::UnetFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::v_unet_mobilenetv2_256x256::UnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
