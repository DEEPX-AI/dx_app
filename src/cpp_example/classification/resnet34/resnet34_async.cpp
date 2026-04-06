/**
 * @file resnet34_async.cpp
 * @brief ResNet34 asynchronous classification example
 */

#include "factory/resnet34_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet34Factory>();
    dxapp::AsyncClassificationRunner<dxapp::ResNet34Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
