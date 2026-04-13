/**
 * @file efficientformer_l7_sync.cpp
 * @brief Efficientformer_l7 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientformer_l7_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientformer_l7Factory>();
    dxapp::SyncClassificationRunner<dxapp::Efficientformer_l7Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
