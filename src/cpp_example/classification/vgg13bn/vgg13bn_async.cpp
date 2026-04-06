/**
 * @file vgg13bn_async.cpp
 * @brief Vgg13bn asynchronous classification example
 */

#include "factory/vgg13bn_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg13bnFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg13bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
