/**
 * @file vgg19_sync.cpp
 * @brief Vgg19 synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg19_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg19Factory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg19Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
