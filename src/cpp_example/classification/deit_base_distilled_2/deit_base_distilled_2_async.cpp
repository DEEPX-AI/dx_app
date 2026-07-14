/**
 * @file deit_base_distilled_async.cpp
 * @brief DeitBaseDistilled2Factory asynchronous inference example
 */

#include "factory/deit_base_distilled_2_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeitBaseDistilled2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::DeitBaseDistilled2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
