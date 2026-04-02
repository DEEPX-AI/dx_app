/**
 * @file mobilenetv3large_sync.cpp
 * @brief Mobilenetv3large synchronous classification example using SyncClassificationRunner
 */

#include "factory/mobilenetv3large_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mobilenetv3largeFactory>();
    dxapp::SyncClassificationRunner<dxapp::Mobilenetv3largeFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
