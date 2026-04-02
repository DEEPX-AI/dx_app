/**
 * @file efficientnetv2s_async.cpp
 * @brief EfficientNetv2s asynchronous classification example
 */

#include "factory/efficientnetv2s_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetv2sFactory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetv2sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
