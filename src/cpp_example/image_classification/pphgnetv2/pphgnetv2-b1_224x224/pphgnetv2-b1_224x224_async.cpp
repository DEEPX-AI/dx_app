/**
 * @file pphgnetv2-b1_224x224
 * @brief pphgnetv2 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/pphgnetv2-b1_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_pphgnetv2_b1_224x224::Pphgnetv2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::v_pphgnetv2_b1_224x224::Pphgnetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
