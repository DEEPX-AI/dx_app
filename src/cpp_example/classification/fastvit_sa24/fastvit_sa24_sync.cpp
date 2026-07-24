/**
 * @file fastvit_sa24_sync.cpp
 * @brief FastvitSa24Factory synchronous inference example
 */

#include "factory/fastvit_sa24_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::FastvitSa24Factory>();
    dxapp::SyncClassificationRunner<dxapp::FastvitSa24Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
