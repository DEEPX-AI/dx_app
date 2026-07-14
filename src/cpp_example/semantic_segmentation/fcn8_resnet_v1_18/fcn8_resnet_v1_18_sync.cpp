/**
 * @file fcn8_resnet18_sync.cpp
 * @brief Fcn8Resnet18Factory synchronous inference example
 */

#include "factory/fcn8_resnet_v1_18_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Fcn8Resnet18Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::Fcn8Resnet18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
