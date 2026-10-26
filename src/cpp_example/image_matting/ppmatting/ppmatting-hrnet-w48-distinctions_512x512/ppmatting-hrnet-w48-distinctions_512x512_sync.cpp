/**
 * @file ppmatting-hrnet-w48-distinctions_512x512
 * @brief ppmatting sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ppmatting-hrnet-w48-distinctions_512x512_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ppmatting_hrnet_w48_distinctions_512x512::PpmattingFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::v_ppmatting_hrnet_w48_distinctions_512x512::PpmattingFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
