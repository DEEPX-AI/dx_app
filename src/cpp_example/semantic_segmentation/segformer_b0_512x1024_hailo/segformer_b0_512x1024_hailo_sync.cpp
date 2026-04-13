/**
 * @file segformer_b0_512x1024_hailo_sync.cpp
 * @brief Segformer_b0_512x1024_hailo synchronous semantic segmentation example
 */

#include "factory/segformer_b0_512x1024_hailo_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Segformer_b0_512x1024_hailoFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::Segformer_b0_512x1024_hailoFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
