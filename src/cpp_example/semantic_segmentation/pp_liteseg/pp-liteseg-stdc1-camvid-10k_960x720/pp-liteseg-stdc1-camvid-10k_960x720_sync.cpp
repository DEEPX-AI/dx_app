/**
 * @file pp-liteseg-stdc1-camvid-10k_960x720
 * @brief pp_liteseg sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/pp-liteseg-stdc1-camvid-10k_960x720_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_pp_liteseg_stdc1_camvid_10k_960x720::PpLitesegFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::v_pp_liteseg_stdc1_camvid_10k_960x720::PpLitesegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
