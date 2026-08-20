/**
 * @file yolo26_depth_x_async.cpp
 * @brief Yolo26DepthXFactory asynchronous depth estimation example
 */

#include "factory/yolo26_depth_x_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthXFactory>();
    dxapp::AsyncDepthRunner<dxapp::Yolo26DepthXFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
