/**
 * @file deeplabv3plus_drn_512x512_async.cpp
 * @brief DeepLabV3+ DRN 512x512 asynchronous inference example
 */

#include "factory/deeplabv3plus_drn_512x512_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deeplabv3plusDrn512x512Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::Deeplabv3plusDrn512x512Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
