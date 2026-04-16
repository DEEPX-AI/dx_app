/**
 * @file regnetx1_6gf_h_sync.cpp
 * @brief Regnetx1_6gf_h synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx1_6gf_h_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx1_6gf_hFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx1_6gf_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
