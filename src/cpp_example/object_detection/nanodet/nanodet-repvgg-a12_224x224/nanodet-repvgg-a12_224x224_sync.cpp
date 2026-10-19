/**
 * @file nanodet-repvgg-a12_224x224
 * @brief nanodet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/nanodet-repvgg-a12_224x224_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_nanodet_repvgg_a12_224x224::NanodetFactory>();
    dxapp::SyncDetectionRunner<dxapp::v_nanodet_repvgg_a12_224x224::NanodetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
