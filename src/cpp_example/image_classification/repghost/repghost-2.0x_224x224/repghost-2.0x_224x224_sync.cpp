/**
 * @file repghost-2.0x_224x224
 * @brief repghost sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/repghost-2.0x_224x224_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_repghost_2_0x_224x224::RepghostFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_repghost_2_0x_224x224::RepghostFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
