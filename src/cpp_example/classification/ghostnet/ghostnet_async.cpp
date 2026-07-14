/**
 * @file ghostnet_async.cpp
 * @brief GhostnetFactory asynchronous inference example
 */

#include "factory/ghostnet_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::GhostnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::GhostnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
