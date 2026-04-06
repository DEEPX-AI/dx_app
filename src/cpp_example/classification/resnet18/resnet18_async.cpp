/**
 * @file resnet18_async.cpp
 * @brief ResNet18 asynchronous classification example
 */

#include "factory/resnet18_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet18Factory>();
    dxapp::AsyncClassificationRunner<dxapp::ResNet18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
