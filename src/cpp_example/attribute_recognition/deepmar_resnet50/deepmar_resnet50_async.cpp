/**
 * @file deepmar_resnet50_async.cpp
 * @brief Deepmar_ResNet50 asynchronous classification example
 */

#include "factory/deepmar_resnet50_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deepmar_ResNet50Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Deepmar_ResNet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
