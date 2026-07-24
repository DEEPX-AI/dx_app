/**
 * @file repvgg_b0_async.cpp
 * @brief RepvggB0Factory asynchronous inference example
 */

#include "factory/repvgg_b0_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggB0Factory>();
    dxapp::AsyncClassificationRunner<dxapp::RepvggB0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
