/**
 * @file bisenetv1_sync.cpp
 * @brief BiseNetV1 synchronous semantic segmentation example
 */

#include "factory/bisenetv1_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::BiseNetV1Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::BiseNetV1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
