/**
 * @file efficientnet_lite3_async.cpp
 * @brief EfficientNet_lite3 asynchronous classification example
 */

#include "factory/efficientnet_lite3_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite3Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNet_lite3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
