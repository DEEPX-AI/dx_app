/**
 * @file deeplabv3_resnet101_async.cpp
 * @brief Deeplabv3Resnet101Factory asynchronous inference example
 */

#include "factory/deeplabv3_resnet101_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deeplabv3Resnet101Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::Deeplabv3Resnet101Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
