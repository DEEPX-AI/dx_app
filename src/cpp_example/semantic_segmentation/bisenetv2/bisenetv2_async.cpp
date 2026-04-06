/**
 * @file bisenetv2_async.cpp
 * @brief BiseNetV2 asynchronous semantic segmentation example
 */

#include "factory/bisenetv2_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::BiseNetV2Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::BiseNetV2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
