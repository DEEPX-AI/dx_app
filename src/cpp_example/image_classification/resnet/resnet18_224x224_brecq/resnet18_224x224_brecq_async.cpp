/**
 * @file resnet18_224x224_brecq
 * @brief resnet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/resnet18_224x224_brecq_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_resnet18_224x224_brecq::ResnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_resnet18_224x224_brecq::ResnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
