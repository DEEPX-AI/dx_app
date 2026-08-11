/**
 * @file yolo26_depth_m_async.cpp
 * @brief Yolo26DepthMFactory asynchronous depth estimation example
 */

#include "factory/yolo26_depth_m_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthMFactory>();
    dxapp::AsyncDepthRunner<dxapp::Yolo26DepthMFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
