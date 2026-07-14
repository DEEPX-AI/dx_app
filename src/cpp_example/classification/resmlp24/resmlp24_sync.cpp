/**
 * @file resmlp24_sync.cpp
 * @brief Resmlp24Factory synchronous inference example
 */

#include "factory/resmlp24_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resmlp24Factory>();
    dxapp::SyncClassificationRunner<dxapp::Resmlp24Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
