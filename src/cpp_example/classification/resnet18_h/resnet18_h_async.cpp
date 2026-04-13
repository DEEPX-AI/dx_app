/**
 * @file resnet18_h_async.cpp
 * @brief ResNet18_h asynchronous classification example
 */

#include "factory/resnet18_h_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet18_hFactory>();
    dxapp::AsyncClassificationRunner<dxapp::ResNet18_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
