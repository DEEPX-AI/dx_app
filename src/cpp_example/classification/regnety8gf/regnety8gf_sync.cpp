/**
 * @file regnety8gf_sync.cpp
 * @brief Regnety8gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety8gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety8gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety8gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
