/**
 * @file deeplabv3plus-resnet101_512x512
 * @brief deeplabv3 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/deeplabv3plus-resnet101_512x512_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_deeplabv3plus_resnet101_512x512::Deeplabv3Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::v_deeplabv3plus_resnet101_512x512::Deeplabv3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
