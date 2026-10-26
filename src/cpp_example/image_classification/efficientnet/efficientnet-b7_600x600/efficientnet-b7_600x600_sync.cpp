/**
 * @file efficientnet-b7_600x600
 * @brief efficientnet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/efficientnet-b7_600x600_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_efficientnet_b7_600x600::EfficientnetFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_efficientnet_b7_600x600::EfficientnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
