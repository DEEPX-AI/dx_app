/**
 * @file regnety16gf_sync.cpp
 * @brief Regnety16gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety16gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety16gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety16gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
