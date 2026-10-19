/**
 * @file densenet201_224x224
 * @brief densenet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/densenet201_224x224_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_densenet201_224x224::DensenetFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_densenet201_224x224::DensenetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
