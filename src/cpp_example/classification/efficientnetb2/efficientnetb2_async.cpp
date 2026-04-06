/**
 * @file efficientnetb2_async.cpp
 * @brief EfficientNetb2 asynchronous classification example
 */

#include "factory/efficientnetb2_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetb2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
