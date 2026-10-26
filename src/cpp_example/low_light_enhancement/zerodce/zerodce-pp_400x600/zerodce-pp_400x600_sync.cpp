/**
 * @file zerodce-pp_400x600
 * @brief zerodce sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/zerodce-pp_400x600_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_zerodce_pp_400x600::ZerodceFactory>();
    dxapp::SyncRestorationRunner<dxapp::v_zerodce_pp_400x600::ZerodceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
