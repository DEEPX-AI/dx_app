/**
 * @file nanodet-plus-1.5x_224x224
 * @brief nanodet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/nanodet-plus-1.5x_224x224_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_nanodet_plus_1_5x_224x224::NanodetFactory>();
    dxapp::AsyncDetectionRunner<dxapp::v_nanodet_plus_1_5x_224x224::NanodetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
