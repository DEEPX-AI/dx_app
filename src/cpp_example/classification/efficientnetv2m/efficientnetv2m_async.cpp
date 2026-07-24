/**
 * @file efficientnetv2m_async.cpp
 * @brief Efficientnetv2mFactory asynchronous inference example
 */

#include "factory/efficientnetv2m_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientnetv2mFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Efficientnetv2mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
