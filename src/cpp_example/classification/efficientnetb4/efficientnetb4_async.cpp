/**
 * @file efficientnetb4_async.cpp
 * @brief EfficientNetb4 asynchronous classification example
 */

#include "factory/efficientnetb4_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb4Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetb4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
