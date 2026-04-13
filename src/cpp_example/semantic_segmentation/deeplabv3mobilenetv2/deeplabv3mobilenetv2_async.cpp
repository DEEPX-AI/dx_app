/**
 * @file deeplabv3mobilenetv2_async.cpp
 * @brief DeepLabv3mobilenetv2 asynchronous semantic segmentation example
 */

#include "factory/deeplabv3mobilenetv2_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeepLabv3mobilenetv2Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::DeepLabv3mobilenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
