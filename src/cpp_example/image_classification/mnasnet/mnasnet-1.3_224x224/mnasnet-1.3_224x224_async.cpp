/**
 * @file mnasnet-1.3_224x224
 * @brief mnasnet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/mnasnet-1.3_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_mnasnet_1_3_224x224::MnasnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_mnasnet_1_3_224x224::MnasnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
