/**
 * @file deeplabv3plus_resnet50_sync.cpp
 * @brief Deeplabv3plusResnet50Factory synchronous inference example
 */

#include "factory/deeplabv3plus_resnet50_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deeplabv3plusResnet50Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::Deeplabv3plusResnet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
