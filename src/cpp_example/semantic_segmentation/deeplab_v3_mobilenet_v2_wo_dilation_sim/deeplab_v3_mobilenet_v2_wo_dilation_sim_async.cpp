/**
 * @file deeplabv3_mobilenetv2_async.cpp
 * @brief Deeplabv3Mobilenetv2Factory asynchronous inference example
 */

#include "factory/deeplab_v3_mobilenet_v2_wo_dilation_sim_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deeplabv3Mobilenetv2Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::Deeplabv3Mobilenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
