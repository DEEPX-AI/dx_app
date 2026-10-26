/**
 * @file pplcnetv2-base_224x224
 * @brief pplcnetv2 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/pplcnetv2-base_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_pplcnetv2_base_224x224::Pplcnetv2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::v_pplcnetv2_base_224x224::Pplcnetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
