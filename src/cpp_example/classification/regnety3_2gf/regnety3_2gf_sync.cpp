/**
 * @file regnety3_2gf_sync.cpp
 * @brief Regnety3_2gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety3_2gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety3_2gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety3_2gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
