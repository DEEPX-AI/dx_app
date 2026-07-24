/**
 * @file beit_large_p16_async.cpp
 * @brief BeitLargeP16Factory asynchronous inference example
 */

#include "factory/beit_large_patch16_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::BeitLargeP16Factory>();
    dxapp::AsyncClassificationRunner<dxapp::BeitLargeP16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
