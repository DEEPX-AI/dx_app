/**
 * @file pidnet_s_sync.cpp
 * @brief PidnetSFactory synchronous inference example
 */

#include "factory/pidnet_s_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::PidnetSFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::PidnetSFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
