/**
 * @file zerodce_400x600
 * @brief zerodce async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/zerodce_400x600_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_zerodce_400x600::ZerodceFactory>();
    dxapp::AsyncRestorationRunner<dxapp::v_zerodce_400x600::ZerodceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
