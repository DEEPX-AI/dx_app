/**
 * @file resnet50_async.cpp
 * @brief ResNet50 asynchronous classification example
 */

#include "factory/resnet50_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet50Factory>();
    dxapp::AsyncClassificationRunner<dxapp::ResNet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
