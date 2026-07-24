/**
 * @file regnetx1_6gf_v2_sync.cpp
 * @brief Regnetx1_6gf_v2 synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx1_6gf_v2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx1_6gf_v2Factory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx1_6gf_v2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
