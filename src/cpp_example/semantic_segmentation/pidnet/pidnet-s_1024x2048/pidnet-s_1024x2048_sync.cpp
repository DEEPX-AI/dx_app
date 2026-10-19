/**
 * @file pidnet-s_1024x2048
 * @brief pidnet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/pidnet-s_1024x2048_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_pidnet_s_1024x2048::PidnetFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::v_pidnet_s_1024x2048::PidnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
