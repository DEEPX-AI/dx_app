/**
 * @file beit-l-p16_224x224
 * @brief beit async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/beit-l-p16_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_beit_l_p16_224x224::BeitFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_beit_l_p16_224x224::BeitFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
