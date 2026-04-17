/**
 * @file efficientnet_lite4_async.cpp
 * @brief EfficientNet_lite4 asynchronous classification example
 */

#include "factory/efficientnet_lite4_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite4Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNet_lite4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
