/**
 * @file yolo26_depth_l_async.cpp
 * @brief Yolo26DepthLFactory asynchronous depth estimation example
 */

#include "factory/yolo26_depth_l_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthLFactory>();
    dxapp::AsyncDepthRunner<dxapp::Yolo26DepthLFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
