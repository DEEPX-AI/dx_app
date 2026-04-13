/**
 * @file efficientformer_l3_sync.cpp
 * @brief Efficientformer_l3 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientformer_l3_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientformer_l3Factory>();
    dxapp::SyncClassificationRunner<dxapp::Efficientformer_l3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
