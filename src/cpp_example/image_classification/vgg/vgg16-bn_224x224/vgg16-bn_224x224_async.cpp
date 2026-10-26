/**
 * @file vgg16-bn_224x224
 * @brief vgg async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/vgg16-bn_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_vgg16_bn_224x224::VggFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_vgg16_bn_224x224::VggFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
