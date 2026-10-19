/**
 * @file efficientnetv2-l_480x480
 * @brief efficientnet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/efficientnetv2-l_480x480_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_efficientnetv2_l_480x480::EfficientnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_efficientnetv2_l_480x480::EfficientnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
