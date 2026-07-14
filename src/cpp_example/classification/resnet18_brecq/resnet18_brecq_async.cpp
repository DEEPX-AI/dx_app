/**
 * @file resnet18_brecq_async.cpp
 * @brief ResNet18_brecq asynchronous classification example
 */

#include "factory/resnet18_brecq_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet18_brecqFactory>();
    dxapp::AsyncClassificationRunner<dxapp::ResNet18_brecqFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
