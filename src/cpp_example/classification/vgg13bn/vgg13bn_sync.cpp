/**
 * @file vgg13bn_sync.cpp
 * @brief Vgg13bn synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg13bn_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg13bnFactory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg13bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
