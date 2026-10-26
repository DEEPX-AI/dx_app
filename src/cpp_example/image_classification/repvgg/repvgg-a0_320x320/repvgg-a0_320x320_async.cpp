/**
 * @file repvgg-a0_320x320
 * @brief repvgg async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/repvgg-a0_320x320_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_repvgg_a0_320x320::RepvggFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_repvgg_a0_320x320::RepvggFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
