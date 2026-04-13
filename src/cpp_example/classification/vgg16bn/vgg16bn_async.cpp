/**
 * @file vgg16bn_async.cpp
 * @brief Vgg16bn asynchronous classification example
 */

#include "factory/vgg16bn_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg16bnFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg16bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
