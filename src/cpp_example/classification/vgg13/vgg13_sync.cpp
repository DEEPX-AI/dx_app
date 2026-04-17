/**
 * @file vgg13_sync.cpp
 * @brief Vgg13 synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg13_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg13Factory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg13Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
