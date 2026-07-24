/**
 * @file fastvit_ma36_sync.cpp
 * @brief FastvitMa36Factory synchronous inference example
 */

#include "factory/fastvit_ma36_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::FastvitMa36Factory>();
    dxapp::SyncClassificationRunner<dxapp::FastvitMa36Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
