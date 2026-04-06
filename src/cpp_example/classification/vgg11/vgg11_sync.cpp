/**
 * @file vgg11_sync.cpp
 * @brief Vgg11 synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg11_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg11Factory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg11Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
