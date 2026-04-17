/**
 * @file efficientnetb3_async.cpp
 * @brief EfficientNetb3 asynchronous classification example
 */

#include "factory/efficientnetb3_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb3Factory>();
    dxapp::AsyncClassificationRunner<dxapp::EfficientNetb3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
