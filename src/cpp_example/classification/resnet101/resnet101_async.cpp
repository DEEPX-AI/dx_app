/**
 * @file resnet101_async.cpp
 * @brief ResNet101 asynchronous classification example
 */

#include "factory/resnet101_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet101Factory>();
    dxapp::AsyncClassificationRunner<dxapp::ResNet101Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
