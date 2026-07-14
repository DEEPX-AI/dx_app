/**
 * @file repvgg_b1_sync.cpp
 * @brief RepvggB1Factory synchronous inference example
 */

#include "factory/repvgg_b1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggB1Factory>();
    dxapp::SyncClassificationRunner<dxapp::RepvggB1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
