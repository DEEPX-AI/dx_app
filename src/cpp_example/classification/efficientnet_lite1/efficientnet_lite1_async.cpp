/**
 * @file efficientnet_lite1_async.cpp
 * @brief EfficientNet_lite1 asynchronous classification example
 */

#include "factory/efficientnet_lite1_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite1Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNet_lite1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
