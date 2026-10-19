/**
 * @file efficientnet-lite2_260x260
 * @brief efficientnet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/efficientnet-lite2_260x260_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_efficientnet_lite2_260x260::EfficientnetFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_efficientnet_lite2_260x260::EfficientnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
