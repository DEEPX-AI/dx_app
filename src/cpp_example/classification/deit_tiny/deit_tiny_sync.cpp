/**
 * @file deit_tiny_sync.cpp
 * @brief DeitTinyFactory synchronous inference example
 */

#include "factory/deit_tiny_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitTinyFactory>();
    dxapp::SyncClassificationRunner<dxapp::DeitTinyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
