/**
 * @file deit-b_224x224_distilled
 * @brief deit async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/deit-b_224x224_distilled_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_deit_b_224x224_distilled::DeitFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_deit_b_224x224_distilled::DeitFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
