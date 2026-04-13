/**
 * @file unet_mobilenet_v2_async.cpp
 * @brief Unet_mobilenet_v2 asynchronous semantic segmentation example
 */

#include "factory/unet_mobilenet_v2_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Unet_mobilenet_v2Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::Unet_mobilenet_v2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
