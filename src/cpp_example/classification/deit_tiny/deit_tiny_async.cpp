/**
 * @file deit_tiny_async.cpp
 * @brief DeitTinyFactory asynchronous inference example
 */

#include "factory/deit_tiny_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitTinyFactory>();
    dxapp::AsyncClassificationRunner<dxapp::DeitTinyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
