/**
 * @file patchcore_224x224
 * @brief patchcore sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/patchcore_224x224_factory.hpp"
#include "common/runner/sync_anomaly_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_patchcore_224x224::PatchcoreFactory>();
    dxapp::SyncAnomalyRunner<dxapp::v_patchcore_224x224::PatchcoreFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
