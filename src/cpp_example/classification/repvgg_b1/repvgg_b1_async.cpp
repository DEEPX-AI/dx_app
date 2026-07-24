/**
 * @file repvgg_b1_async.cpp
 * @brief RepvggB1Factory asynchronous inference example
 */

#include "factory/repvgg_b1_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RepvggB1Factory>();
    dxapp::AsyncClassificationRunner<dxapp::RepvggB1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
