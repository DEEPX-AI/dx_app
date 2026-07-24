/**
 * @file fcn8_resnet18_async.cpp
 * @brief Fcn8Resnet18Factory asynchronous inference example
 */

#include "factory/fcn8_resnet_v1_18_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Fcn8Resnet18Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::Fcn8Resnet18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
