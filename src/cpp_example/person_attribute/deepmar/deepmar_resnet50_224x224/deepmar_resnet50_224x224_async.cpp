/**
 * @file deepmar_resnet50_224x224
 * @brief deepmar async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/deepmar_resnet50_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_deepmar_resnet50_224x224::DeepmarFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_deepmar_resnet50_224x224::DeepmarFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
