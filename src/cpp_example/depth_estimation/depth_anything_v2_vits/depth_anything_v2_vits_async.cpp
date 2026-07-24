/**
 * @file depth_anything_v2_vits_async.cpp
 * @brief DepthAnythingV2VitsFactory asynchronous depth estimation example
 */

#include "factory/depth_anything_v2_vits_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DepthAnythingV2VitsFactory>();
    dxapp::AsyncDepthRunner<dxapp::DepthAnythingV2VitsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
