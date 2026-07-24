/**
 * @file resnext101_32x8d_async.cpp
 * @brief Resnext10132x8dFactory asynchronous inference example
 */

#include "factory/resnext101_32x8d_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resnext10132x8dFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Resnext10132x8dFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
