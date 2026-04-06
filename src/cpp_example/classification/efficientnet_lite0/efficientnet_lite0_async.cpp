/**
 * @file efficientnet_lite0_async.cpp
 * @brief EfficientNet asynchronous classification example
 */

#include "factory/efficientnet_lite0_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
