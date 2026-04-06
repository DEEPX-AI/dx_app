/**
 * @file bisenetv1_async.cpp
 * @brief BiseNetV1 asynchronous semantic segmentation example
 */

#include "factory/bisenetv1_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::BiseNetV1Factory>();
    dxapp::AsyncSemanticSegRunner<dxapp::BiseNetV1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
