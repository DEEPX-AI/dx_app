/**
 * @file shufflenetv1-x1.0_224x224
 * @brief shufflenetv1 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/shufflenetv1-x1.0_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_shufflenetv1_x1_0_224x224::Shufflenetv1Factory>();
    dxapp::AsyncClassificationRunner<dxapp::v_shufflenetv1_x1_0_224x224::Shufflenetv1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
