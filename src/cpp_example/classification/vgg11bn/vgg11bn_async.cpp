/**
 * @file vgg11bn_async.cpp
 * @brief Vgg11bn asynchronous classification example
 */

#include "factory/vgg11bn_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vgg11bnFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Vgg11bnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
