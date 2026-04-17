/**
 * @file segformer_b0_512x1024_async.cpp
 * @brief SegFormer-B0 asynchronous semantic segmentation example
 */

#include "factory/segformer_b0_512x1024_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SegFormerFactory>();
    dxapp::AsyncSemanticSegRunner<dxapp::SegFormerFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
