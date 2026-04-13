/**
 * @file vgg16_sync.cpp
 * @brief Vgg16 synchronous classification example using SyncClassificationRunner
 */

#include "factory/vgg16_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg16Factory>();
    dxapp::SyncClassificationRunner<dxapp::Vgg16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
