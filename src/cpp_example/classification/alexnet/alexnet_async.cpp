/**
 * @file alexnet_async.cpp
 * @brief Alexnet asynchronous classification example
 */

#include "factory/alexnet_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::AlexnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::AlexnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
