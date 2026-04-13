/**
 * @file resnet152_async.cpp
 * @brief ResNet152 asynchronous classification example
 */

#include "factory/resnet152_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet152Factory>();
    dxapp::AsyncClassificationRunner<dxapp::ResNet152Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
