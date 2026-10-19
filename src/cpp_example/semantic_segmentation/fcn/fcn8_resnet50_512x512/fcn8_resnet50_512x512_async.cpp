/**
 * @file fcn8_resnet50_512x512
 * @brief fcn async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/fcn8_resnet50_512x512_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_fcn8_resnet50_512x512::FcnFactory>();
    dxapp::AsyncSemanticSegRunner<dxapp::v_fcn8_resnet50_512x512::FcnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
