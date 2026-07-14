/**
 * @file deit_base384_distilled_async.cpp
 * @brief DeitBase384DistilledFactory asynchronous inference example
 */

#include "factory/deit_base384_distilled_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitBase384DistilledFactory>();
    dxapp::AsyncClassificationRunner<dxapp::DeitBase384DistilledFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
