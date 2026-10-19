/**
 * @file stdc1-seg50_512x1024
 * @brief stdc_seg sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/stdc1-seg50_512x1024_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_stdc1_seg50_512x1024::StdcSegFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::v_stdc1_seg50_512x1024::StdcSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
