/**
 * @file segformer_mit-b0_512x1024
 * @brief segformer async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/segformer_mit-b0_512x1024_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_segformer_mit_b0_512x1024::SegformerFactory>();
    dxapp::AsyncSemanticSegRunner<dxapp::v_segformer_mit_b0_512x1024::SegformerFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
