/**
 * @file dncnn-50_512x512
 * @brief dncnn sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/dncnn-50_512x512_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_dncnn_50_512x512::DncnnFactory>();
    dxapp::SyncRestorationRunner<dxapp::v_dncnn_50_512x512::DncnnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
