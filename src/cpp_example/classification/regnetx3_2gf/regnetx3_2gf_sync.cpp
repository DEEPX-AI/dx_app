/**
 * @file regnetx3_2gf_sync.cpp
 * @brief Regnetx3_2gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx3_2gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx3_2gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx3_2gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
