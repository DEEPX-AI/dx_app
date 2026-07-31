/**
 * @file yolo26n_depth_async.cpp
 * @brief Yolo26nDepthFactory asynchronous depth estimation example
 */

#include "factory/yolo26n_depth_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26nDepthFactory>();
    dxapp::AsyncDepthRunner<dxapp::Yolo26nDepthFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
