/**
 * @file repvgg_a0_async.cpp
 * @brief RepvggA0Factory asynchronous inference example
 */

#include "factory/repvgg_a0_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggA0Factory>();
    dxapp::AsyncClassificationRunner<dxapp::RepvggA0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
