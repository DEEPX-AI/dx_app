/**
 * @file repghost_2_0x_async.cpp
 * @brief Repghost20xFactory asynchronous inference example
 */

#include "factory/repghost_2_0x_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Repghost20xFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Repghost20xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
