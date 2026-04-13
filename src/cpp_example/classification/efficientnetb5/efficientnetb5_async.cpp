/**
 * @file efficientnetb5_async.cpp
 * @brief EfficientNetb5 asynchronous classification example
 */

#include "factory/efficientnetb5_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb5Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetb5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
