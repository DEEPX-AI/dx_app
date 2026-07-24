/**
 * @file repvgg_a1_async.cpp
 * @brief RepvggA1Factory asynchronous inference example
 */

#include "factory/repvgg_a1_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggA1Factory>();
    dxapp::AsyncClassificationRunner<dxapp::RepvggA1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
