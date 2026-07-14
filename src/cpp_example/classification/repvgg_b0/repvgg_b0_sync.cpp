/**
 * @file repvgg_b0_sync.cpp
 * @brief RepvggB0Factory synchronous inference example
 */

#include "factory/repvgg_b0_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggB0Factory>();
    dxapp::SyncClassificationRunner<dxapp::RepvggB0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
