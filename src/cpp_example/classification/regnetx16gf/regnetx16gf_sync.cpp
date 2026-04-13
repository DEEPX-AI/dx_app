/**
 * @file regnetx16gf_sync.cpp
 * @brief Regnetx16gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx16gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx16gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx16gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
