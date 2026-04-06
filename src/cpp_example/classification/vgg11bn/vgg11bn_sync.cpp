/**
 * @file vgg11bn_sync.cpp
 * @brief Vgg11bn synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg11bn_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg11bnFactory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg11bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
