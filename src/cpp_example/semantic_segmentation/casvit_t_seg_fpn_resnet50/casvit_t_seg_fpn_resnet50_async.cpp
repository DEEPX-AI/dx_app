/**
 * @file casvit_t_seg_fpn_resnet50_async.cpp
 * @brief CasvitTSegFpnResnet50Factory asynchronous inference example
 */

#include "factory/casvit_t_seg_fpn_resnet50_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::CasvitTSegFpnResnet50Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::CasvitTSegFpnResnet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
