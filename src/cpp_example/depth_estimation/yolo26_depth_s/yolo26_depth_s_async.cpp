/**
 * @file yolo26_depth_s_async.cpp
 * @brief Yolo26DepthSFactory asynchronous depth estimation example
 */

#include "factory/yolo26_depth_s_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthSFactory>();
    dxapp::AsyncDepthRunner<dxapp::Yolo26DepthSFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
