/**
 * @file deeplabv3plusmobilenetv2_async.cpp
 * @brief DeepLabv3plusmobilenetv2 asynchronous semantic segmentation example
 */

#include "factory/deeplabv3plusmobilenetv2_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeepLabv3plusmobilenetv2Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::DeepLabv3plusmobilenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
