/**
 * @file vgg13_async.cpp
 * @brief Vgg13 asynchronous classification example
 */

#include "factory/vgg13_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg13Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg13Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
