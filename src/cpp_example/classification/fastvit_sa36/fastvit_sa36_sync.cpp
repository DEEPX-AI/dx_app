/**
 * @file fastvit_sa36_sync.cpp
 * @brief FastvitSa36Factory synchronous inference example
 */

#include "factory/fastvit_sa36_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::FastvitSa36Factory>();
    dxapp::SyncClassificationRunner<dxapp::FastvitSa36Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
