/**
 * @file deit_small_distilled_async.cpp
 * @brief DeitSmallDistilledFactory asynchronous inference example
 */

#include "factory/deit_small_distilled_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitSmallDistilledFactory>();
    dxapp::AsyncClassificationRunner<dxapp::DeitSmallDistilledFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
