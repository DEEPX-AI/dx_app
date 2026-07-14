/**
 * @file repvgg_a2_sync.cpp
 * @brief RepvggA2Factory synchronous inference example
 */

#include "factory/repvgg_a2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggA2Factory>();
    dxapp::SyncClassificationRunner<dxapp::RepvggA2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
