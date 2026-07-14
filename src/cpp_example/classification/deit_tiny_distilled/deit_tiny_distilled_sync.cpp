/**
 * @file deit_tiny_distilled_sync.cpp
 * @brief DeitTinyDistilledFactory synchronous inference example
 */

#include "factory/deit_tiny_distilled_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitTinyDistilledFactory>();
    dxapp::SyncClassificationRunner<dxapp::DeitTinyDistilledFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
