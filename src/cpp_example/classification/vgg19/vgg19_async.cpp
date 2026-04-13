/**
 * @file vgg19_async.cpp
 * @brief Vgg19 asynchronous classification example
 */

#include "factory/vgg19_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg19Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg19Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
