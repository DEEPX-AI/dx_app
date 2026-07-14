/**
 * @file vit_base_bn_async.cpp
 * @brief VitBaseBnFactory asynchronous inference example
 */

#include "factory/vit_base_bn_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitBaseBnFactory>();
    dxapp::AsyncClassificationRunner<dxapp::VitBaseBnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
