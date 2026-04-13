/**
 * @file efficientnetb6_async.cpp
 * @brief EfficientNetb6 asynchronous classification example
 */

#include "factory/efficientnetb6_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb6Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetb6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
