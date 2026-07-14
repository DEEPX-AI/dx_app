/**
 * @file deit_base_async.cpp
 * @brief DeitBaseFactory asynchronous inference example
 */

#include "factory/deit_base_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitBaseFactory>();
    dxapp::AsyncClassificationRunner<dxapp::DeitBaseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
