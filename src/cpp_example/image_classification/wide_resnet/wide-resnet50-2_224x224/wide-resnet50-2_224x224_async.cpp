/**
 * @file wide-resnet50-2_224x224
 * @brief wide_resnet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/wide-resnet50-2_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_wide_resnet50_2_224x224::WideResnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_wide_resnet50_2_224x224::WideResnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
