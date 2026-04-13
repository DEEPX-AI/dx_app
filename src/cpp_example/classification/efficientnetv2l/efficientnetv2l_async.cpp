/**
 * @file efficientnetv2l_async.cpp
 * @brief EfficientNetv2l asynchronous classification example
 */

#include "factory/efficientnetv2l_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetv2lFactory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetv2lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
