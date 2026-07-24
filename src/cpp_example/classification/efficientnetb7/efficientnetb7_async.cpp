/**
 * @file efficientnetb7_async.cpp
 * @brief Efficientnetb7Factory asynchronous inference example
 */

#include "factory/efficientnetb7_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientnetb7Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Efficientnetb7Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
