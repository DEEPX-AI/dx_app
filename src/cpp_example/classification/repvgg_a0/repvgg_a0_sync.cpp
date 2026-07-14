/**
 * @file repvgg_a0_sync.cpp
 * @brief RepvggA0Factory synchronous inference example
 */

#include "factory/repvgg_a0_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggA0Factory>();
    dxapp::SyncClassificationRunner<dxapp::RepvggA0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
