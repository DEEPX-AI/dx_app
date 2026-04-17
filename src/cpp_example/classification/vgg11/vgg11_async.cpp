/**
 * @file vgg11_async.cpp
 * @brief Vgg11 asynchronous classification example
 */

#include "factory/vgg11_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg11Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg11Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
