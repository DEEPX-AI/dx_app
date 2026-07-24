/**
 * @file fastvit_sa36_async.cpp
 * @brief FastvitSa36Factory asynchronous inference example
 */

#include "factory/fastvit_sa36_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::FastvitSa36Factory>();
    dxapp::AsyncClassificationRunner<dxapp::FastvitSa36Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
