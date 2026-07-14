/**
 * @file fastvit_sa24_async.cpp
 * @brief FastvitSa24Factory asynchronous inference example
 */

#include "factory/fastvit_sa24_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::FastvitSa24Factory>();
    dxapp::AsyncClassificationRunner<dxapp::FastvitSa24Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
