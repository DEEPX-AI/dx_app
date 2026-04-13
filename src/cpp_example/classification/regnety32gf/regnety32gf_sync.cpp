/**
 * @file regnety32gf_sync.cpp
 * @brief Regnety32gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety32gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety32gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety32gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
