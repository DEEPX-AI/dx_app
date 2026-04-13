/**
 * @file vgg16_async.cpp
 * @brief Vgg16 asynchronous classification example
 */

#include "factory/vgg16_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg16Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
