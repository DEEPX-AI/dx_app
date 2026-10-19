/**
 * @file vit-b-p16_224x224_bn
 * @brief vit async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/vit-b-p16_224x224_bn_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_vit_b_p16_224x224_bn::VitFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_vit_b_p16_224x224_bn::VitFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
