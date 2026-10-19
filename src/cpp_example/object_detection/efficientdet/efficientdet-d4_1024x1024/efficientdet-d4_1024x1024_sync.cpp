/**
 * @file efficientdet-d4_1024x1024
 * @brief efficientdet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/efficientdet-d4_1024x1024_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_efficientdet_d4_1024x1024::EfficientdetFactory>();
    dxapp::SyncDetectionRunner<dxapp::v_efficientdet_d4_1024x1024::EfficientdetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
