/**
 * @file repvgg_a2_async.cpp
 * @brief RepvggA2Factory asynchronous inference example
 */

#include "factory/repvgg_a2_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggA2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::RepvggA2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
