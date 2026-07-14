/**
 * @file fcn8_resnet50_sync.cpp
 * @brief Fcn8Resnet50Factory synchronous inference example
 */

#include "factory/fcn8_resnet50_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Fcn8Resnet50Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::Fcn8Resnet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
