/**
 * @file vgg19bn_sync.cpp
 * @brief Vgg19bn synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg19bn_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg19bnFactory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg19bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
