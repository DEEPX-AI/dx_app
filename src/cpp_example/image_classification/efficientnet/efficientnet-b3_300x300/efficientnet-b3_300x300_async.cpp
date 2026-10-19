/**
 * @file efficientnet-b3_300x300
 * @brief efficientnet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/efficientnet-b3_300x300_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_efficientnet_b3_300x300::EfficientnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_efficientnet_b3_300x300::EfficientnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
