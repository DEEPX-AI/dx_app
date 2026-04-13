/**
 * @file regnetx1_6gf_sync.cpp
 * @brief Regnetx1_6gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx1_6gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx1_6gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx1_6gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
