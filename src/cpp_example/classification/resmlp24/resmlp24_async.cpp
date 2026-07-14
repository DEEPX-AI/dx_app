/**
 * @file resmlp24_async.cpp
 * @brief Resmlp24Factory asynchronous inference example
 */

#include "factory/resmlp24_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resmlp24Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Resmlp24Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
