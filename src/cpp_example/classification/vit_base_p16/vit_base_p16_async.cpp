/**
 * @file vit_base_p16_async.cpp
 * @brief VitBaseP16Factory asynchronous inference example
 */

#include "factory/vit_base_p16_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitBaseP16Factory>();
    dxapp::AsyncClassificationRunner<dxapp::VitBaseP16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
