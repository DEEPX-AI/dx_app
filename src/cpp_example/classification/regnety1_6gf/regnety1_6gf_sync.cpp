/**
 * @file regnety1_6gf_sync.cpp
 * @brief Regnety1_6gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety1_6gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety1_6gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety1_6gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
