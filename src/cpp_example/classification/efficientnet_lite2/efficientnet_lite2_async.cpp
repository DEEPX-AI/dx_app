/**
 * @file efficientnet_lite2_async.cpp
 * @brief EfficientNet_lite2 asynchronous classification example
 */

#include "factory/efficientnet_lite2_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNet_lite2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
