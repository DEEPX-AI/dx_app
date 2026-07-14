/**
 * @file deit_tiny_distilled_async.cpp
 * @brief DeitTinyDistilledFactory asynchronous inference example
 */

#include "factory/deit_tiny_distilled_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitTinyDistilledFactory>();
    dxapp::AsyncClassificationRunner<dxapp::DeitTinyDistilledFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
