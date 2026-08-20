/**
 * @file yolo26_depth_n_async.cpp
 * @brief Yolo26DepthNFactory asynchronous depth estimation example
 */

#include "factory/yolo26_depth_n_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthNFactory>();
    dxapp::AsyncDepthRunner<dxapp::Yolo26DepthNFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
