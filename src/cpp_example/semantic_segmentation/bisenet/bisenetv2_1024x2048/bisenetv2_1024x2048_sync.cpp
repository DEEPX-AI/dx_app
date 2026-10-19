/**
 * @file bisenetv2_1024x2048
 * @brief bisenet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/bisenetv2_1024x2048_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_bisenetv2_1024x2048::BisenetFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::v_bisenetv2_1024x2048::BisenetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
