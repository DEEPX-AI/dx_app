/**
 * @file realesrgan-x2_192x192
 * @brief realesrgan sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/realesrgan-x2_192x192_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_realesrgan_x2_192x192::RealesrganFactory>();
    dxapp::SyncRestorationRunner<dxapp::v_realesrgan_x2_192x192::RealesrganFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
