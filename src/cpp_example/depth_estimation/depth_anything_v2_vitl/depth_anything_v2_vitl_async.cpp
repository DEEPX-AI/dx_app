/**
 * @file depth_anything_v2_vitl_async.cpp
 * @brief DepthAnythingV2VitlFactory asynchronous depth estimation example
 */

#include "factory/depth_anything_v2_vitl_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DepthAnythingV2VitlFactory>();
    dxapp::AsyncDepthRunner<dxapp::DepthAnythingV2VitlFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
