/**
 * @file bisenetv2_sync.cpp
 * @brief BiseNetV2 synchronous semantic segmentation example
 */

#include "factory/bisenetv2_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::BiseNetV2Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::BiseNetV2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
