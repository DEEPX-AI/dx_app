/**
 * @file ddrnet23_1024x2048
 * @brief ddrnet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ddrnet23_1024x2048_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ddrnet23_1024x2048::DdrnetFactory>();
    dxapp::AsyncSemanticSegRunner<dxapp::v_ddrnet23_1024x2048::DdrnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
