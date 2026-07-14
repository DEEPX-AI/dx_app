/**
 * @file deit_small_async.cpp
 * @brief DeitSmallFactory asynchronous inference example
 */

#include "factory/deit_small_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitSmallFactory>();
    dxapp::AsyncClassificationRunner<dxapp::DeitSmallFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
