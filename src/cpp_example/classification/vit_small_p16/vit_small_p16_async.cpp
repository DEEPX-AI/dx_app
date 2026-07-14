/**
 * @file vit_small_p16_async.cpp
 * @brief VitSmallP16Factory asynchronous inference example
 */

#include "factory/vit_small_p16_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::VitSmallP16Factory>();
    dxapp::AsyncClassificationRunner<dxapp::VitSmallP16Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
