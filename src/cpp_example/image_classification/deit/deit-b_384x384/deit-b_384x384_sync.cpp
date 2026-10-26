/**
 * @file deit-b_384x384
 * @brief deit sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/deit-b_384x384_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_deit_b_384x384::DeitFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_deit_b_384x384::DeitFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
