/**
 * @file deeplabv3plusmobilenet_async.cpp
 * @brief DeepLabv3 asynchronous semantic segmentation example
 */

#include "factory/deeplabv3plusmobilenet_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DeepLabv3Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::DeepLabv3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
