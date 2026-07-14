/**
 * @file fastvit_ma36_async.cpp
 * @brief FastvitMa36Factory asynchronous inference example
 */

#include "factory/fastvit_ma36_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::FastvitMa36Factory>();
    dxapp::AsyncClassificationRunner<dxapp::FastvitMa36Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
