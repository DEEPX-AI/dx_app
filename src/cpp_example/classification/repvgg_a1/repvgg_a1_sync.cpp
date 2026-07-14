/**
 * @file repvgg_a1_sync.cpp
 * @brief RepvggA1Factory synchronous inference example
 */

#include "factory/repvgg_a1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggA1Factory>();
    dxapp::SyncClassificationRunner<dxapp::RepvggA1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
