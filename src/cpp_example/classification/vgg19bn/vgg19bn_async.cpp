/**
 * @file vgg19bn_async.cpp
 * @brief Vgg19bn asynchronous classification example
 */

#include "factory/vgg19bn_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg19bnFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg19bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
