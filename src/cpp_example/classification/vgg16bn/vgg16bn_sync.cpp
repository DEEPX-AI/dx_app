/**
 * @file vgg16bn_sync.cpp
 * @brief Vgg16bn synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg16bn_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg16bnFactory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg16bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
