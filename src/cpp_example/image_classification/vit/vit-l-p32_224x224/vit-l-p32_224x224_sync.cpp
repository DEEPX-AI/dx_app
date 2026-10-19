/**
 * @file vit-l-p32_224x224
 * @brief vit sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/vit-l-p32_224x224_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_vit_l_p32_224x224::VitFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_vit_l_p32_224x224::VitFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
